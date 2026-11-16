"""Fresh-validation-only support-budget ablations through unchanged engines.

Default execution is user-run. --dry-run and --analyze-only perform no model
calls. The pinned B32 reference supplies both the numerical source and control
measurements; new runs alter only support/pool size and output/selection metadata.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

if __package__:
    from .saved_runs import ROOT, checked, executor, identity, immutable_json, read, record_rows, require, sha
else:
    from saved_runs import ROOT, checked, executor, identity, immutable_json, read, record_rows, require, sha


def reference(spec):
    path = ROOT/spec["reference_report"]
    require(sha(path) == spec["reference_report_sha256"], "Pinned B32 report changed")
    report = read(path)
    c = report["config"]
    stage = "P2-followup-iteration" if spec["stage"] == "p2" else "P3c-offline-factorial"
    require(report["stage"] == stage and report["split"] == "validation" and report["completed"]
            and not report["limited_run"], "Use a full completed validation reference")
    require(c["p1"]["data"].get("dataset_id", "pusht") == spec["dataset"], "Reference task differs")
    require(c["scopes"] == ["last-block"] and c["p1"]["evaluation"]["seeds"] == spec["seeds"],
            "Reference scope or seeds differ")
    adaptation = {**c["p1"]["adaptation"], **c["adaptation"]}
    require(adaptation["support_size"] == spec["reference_size"] == 32, "Expected B32 reference")
    require(adaptation["learning_rate"] == spec["learning_rate"], "Reference LR differs")
    require(c["p1"]["retrieval"]["pool_size"] == 32 and c["p1"]["retrieval"]["max_windows_per_episode"] == 4
            and c["p1"]["retrieval"]["min_start_step_gap"] == 20, "Reference pool/diversity settings differ")
    require(set(spec["methods"]) <= set(c["adaptation"]["methods"]) and "frozen" in spec["methods"], "Invalid method subset")
    require(len(set(spec["support_sizes"])) == len(spec["support_sizes"])
            and all(type(b) is int and b >= 2 for b in spec["support_sizes"]), "Use distinct support sizes >=2")
    require(spec["reference_size"] in spec["support_sizes"], "Keep B32 in the declared comparison")
    if spec["stage"] == "p2":
        s = c["schedules"][spec["schedule"]]
        require(s == {"placement": "each-step", "updates": 1, "persistence": "trajectory"},
                "This pilot fixes temporal-U1 with trajectory persistence")
    else:
        require(c["adaptation"]["updates"] == 1, "P3 pilot requires one initial update")
        require(min(spec["support_sizes"]) >= 2*c["planning"]["provisional_actions"],
                "P3 budgets must cover every action/direction stream")
    query = checked(c["query_dataset"]["manifest"])
    audit = checked(Path(c["query_dataset"]["manifest"]).with_name("audit.json"))
    require(query["external_queries"] and not query["episodes"]["memory"] and audit["passed"]
            and audit["query_manifest"] == query["fingerprint"], "Fresh collection/audit mismatch")
    require(report["cohort"]["raw_start_rows"] == query["evaluation_windows"]["validation"]["raw_start_rows"]
            and report["cohort"]["queries"] == 256 and report["cohort"]["episodes"] == 64,
            "Expected full fresh validation q4 cohort")
    return report


def configurations(spec, source, size, directory):
    p1 = deepcopy(source["config"]["p1"])
    p1["adaptation"]["support_size"] = size
    p1["retrieval"]["pool_size"] = size
    p1_path = directory/"configs"/f"p1-b{size}-{identity(p1)[:16]}.json"
    config = deepcopy(source["config"])
    config.pop("p1")
    config["p1_config"] = str(p1_path)
    config["experiment_dir"] = str(directory)
    config["adaptation"]["support_size"] = size
    config["adaptation"]["methods"] = spec["methods"]
    config["adaptation"]["learning_rate"] = spec["learning_rate"]
    config["evaluation"]["records"] = "compact"
    config["evaluation"]["query_limit"] = None
    if spec["stage"] == "p2":
        config["schedule_names"] = [spec["schedule"]]
        config["schedules"] = {spec["schedule"]: deepcopy(config["schedules"][spec["schedule"]])}
    else:
        config["reuse"] = {}
    config_path = directory/"configs"/f"p{spec['stage'][-1]}-b{size}-{identity(config)[:16]}.json"
    expected = {**deepcopy(config), "p1": deepcopy(p1)}
    return p1_path, p1, config_path, config, expected


def cell_key(row, spec):
    return row["method"] if spec["stage"] == "p2" else row["method"]+"/"+row["proposal_source"]


def collect(report, source, spec, size, *, control=False, expected=None):
    require(report["completed"] and not report["limited_run"] and report["split"] == "validation", "Incomplete/non-validation run")
    for key in ("stage", "code", "model", "p1_prepared_fingerprint"):
        require(report[key] == source[key], f"Reference/run mismatch: {key}")
    require(report["cohort"] == source["cohort"], "Ablation changed query cohort")
    if expected is not None:
        require(report["config"] == expected, "Completed run uses another scientific configuration")
        m = checked(Path(expected["experiment_dir"])/"manifest.json")
        require(m["fingerprint"] == report["manifest_fingerprint"], "Changed stage manifest")
    starts = source["cohort"]["raw_start_rows"]
    q = checked(source["config"]["query_dataset"]["manifest"])
    ids = {start: i for i, start in enumerate(starts)}
    sources = source["config"]["proposals"]["sources"] if spec["stage"] == "p3" else [None]
    keys = [m if s is None else m+"/"+s for m, s in product(spec["methods"], sources)]
    values, visited, count = {}, set(), 0
    for row in record_rows(report):
        count += 1
        if control and (row["method"] not in spec["methods"] or (spec["stage"] == "p2" and row["schedule"] != spec["schedule"])):
            continue
        require(row["method"] in spec["methods"] and row["seed"] in spec["seeds"], "Unexpected support-sweep method/seed")
        if spec["stage"] == "p2":
            require(row["schedule"] == spec["schedule"], "Unexpected update schedule")
        start = row["query"]["raw_observation_rows"][0]
        require(start in ids and row["query"]["collection_manifest"] == q["collection_manifest"], "Different query provenance")
        require(row["target_raw_rows"] == [start+10+5*h for h in range(1, 6)] and all(row["checks"].values()),
                "Target mismatch or failed acceptance flag")
        key = (cell_key(row, spec), start, row["seed"])
        require(key not in visited, "Duplicate ablation row")
        visited.add(key)
        frozen = row["method"] == "frozen"
        a = row["adaptation"]
        if spec["stage"] == "p2":
            updates = 0 if frozen else 5
            require(a["actual_updates"] == updates and math.isclose(a["actual_learning_rate_sum"], updates*spec["learning_rate"], abs_tol=1e-12), "Update/LR mismatch")
            supports = [r["support"] for r in a["rounds"]]
            errors, seconds = row["mse_by_block"], row["total_seconds"]
        else:
            updates = 0 if frozen else 1
            require(a["updates"] == updates, "Update-count mismatch")
            supports = [a["support"]] if updates else []
            require(row["cem_candidate_evaluations"] == 1537, "CEM budget changed")
            errors, seconds = row["recorded_action_mse_by_block"], row["primary_seconds_with_checks"]
        require(a["support_presentations"] == updates*size and len(supports) == updates, "Support/update budget shortfall")
        ess = []
        for support in supports:
            require(len(support) == len({s["transition_id"] for s in support}) == size, "Support-size shortfall/duplicate")
            weights = [s["weight"] for s in support]
            require(all(w > 0 for w in weights) and abs(sum(weights)-1) < 1e-6, "Loss weights not normalized")
            episodes = defaultdict(list)
            for s in support:
                require(s["source_episode_position"] != row["query"]["source_episode_position"], "Query episode leaked into support")
                episodes[s["source_episode_position"]].append(s["start_in_episode"])
            require(all(len(v) <= 4 and all(b-a >= 20 for a, b in zip(sorted(v), sorted(v)[1:])) for v in episodes.values()), "Support diversity changed")
            ess.append(1/sum(w*w for w in weights))
        require(len(errors) == 5 and all(math.isfinite(x) and x >= 0 for x in errors), "Invalid forecast")
        values[key] = {"errors": errors, "seconds": seconds, "ESS": sum(ess)/len(ess) if ess else None,
                       "episode": row["query"]["local_episode_position"],
                       "common_sse": row.get("selected_plan_frozen_model_goal_sse"),
                       "own_sse": row.get("selected_model_goal_sse")}
    require(count == report["record_count"], "Record count differs from report")
    require(visited == set(product(keys, starts, spec["seeds"])), "Incomplete query/method/seed matrix")
    return values


def completed(path, source, spec, size, expected):
    if not path.exists():
        require(not path.parent.exists(), f"Partial run preserved: {path.parent}; use another --output-dir")
        return None
    r = read(path)
    collect(r, source, spec, size, expected=expected)
    return r


def analyze(spec, source, records, directory):
    import csv
    import numpy as np
    queries = source["cohort"]["raw_start_rows"]
    seeds = spec["seeds"]
    reference_values = records[spec["reference_size"]]
    harm_tolerance = source["config"]["p1"]["evaluation"]["harm_tolerance"]
    keys = sorted({key[0] for key in reference_values})
    episodes = np.array([reference_values[keys[0], q, seeds[0]]["episode"] for q in queries])
    groups = [np.flatnonzero(episodes == ep) for ep in np.unique(episodes)]
    draws = np.random.default_rng(20261110).integers(0, len(groups), (10000, len(groups)))

    def statistic(values):
        ep = np.array([values[g].mean() for g in groups])
        return {"mean": float(ep.mean()), "ci95": np.quantile(ep[draws].mean(1), [.025, .975]).tolist()}

    def array(rows, cell, metric):
        return np.array([[rows[cell, q, s][metric] for s in seeds] for q in queries])

    means, per_seed, contrasts, frozen = [], [], [], None
    for size, rows in sorted(records.items()):
        for cell in keys:
            method = cell.split('/')[0]
            frozen_cell = 'frozen' if spec['stage'] == 'p2' else 'frozen/'+cell.split('/')[1]
            errors = array(rows, cell, 'errors')
            f = array(rows, frozen_cell, 'errors')
            if frozen is None:
                frozen = f
            require(np.array_equal(frozen, f), "Frozen forecasts changed with support size")
            reference_errors = array(reference_values, cell, 'errors')
            def metrics(seed_index=None):
                sl = slice(None) if seed_index is None else slice(seed_index,seed_index+1)
                delta = errors[:, sl, -1]-f[:, sl, -1]
                ess = array(rows, cell, 'ESS')
                return {"B": size, "setting": cell, "seed": 'mean' if seed_index is None else seeds[seed_index],
                    "reused_reference":size==spec['reference_size'],
                    **{f'H{h}':float(errors[:,sl,h-1].mean()) for h in (1,3,5)},
                    "path_mse":float(errors[:,sl].mean()), "harm_fraction":float((delta>harm_tolerance).mean()),
                    "positive_harm_mass":float(np.maximum(delta,0).mean()),
                    "seconds":float(array(rows,cell,'seconds')[:,sl].mean()),
                    "effective_sample_size":None if method=='frozen' else float(ess[:,sl].mean()),
                    "common_frozen_sse":None if spec['stage']=='p2' else float(array(rows,cell,'common_sse')[:,sl].mean()),
                    "own_model_sse":None if spec['stage']=='p2' else float(array(rows,cell,'own_sse')[:,sl].mean())}
            means.append(metrics()); per_seed.extend(metrics(i) for i in range(len(seeds)))
            budget_contrast={"B":size,"setting":cell,"contrast":"minus B32",
                "H5_delta":statistic((errors[:,:,-1]-reference_errors[:,:,-1]).mean(1))}
            if spec['stage']=='p3':
                budget_contrast['common_frozen_sse_delta']=statistic(
                    (array(rows,cell,'common_sse')-array(reference_values,cell,'common_sse')).mean(1))
            contrasts.append(budget_contrast)
            contrasts.append({"B":size,"setting":cell,"contrast":"minus frozen",
                "H5_delta":statistic((errors[:,:,-1]-f[:,:,-1]).mean(1))})
            if method=='fr-fixed':
                suffix='' if spec['stage']=='p2' else '/'+cell.split('/')[1]
                for control in ('uniform','forward','reverse'):
                    if control+suffix in keys:
                        contrasts.append({"B":size,"setting":cell,"contrast":"minus "+control,
                            "H5_delta":statistic((errors[:,:,-1]-array(rows,control+suffix,'errors')[:,:,-1]).mean(1))})
    directory.mkdir(parents=True,exist_ok=True)
    result={"split":"validation","dataset":spec['dataset'],"stage":spec['stage'],"LR":spec['learning_rate'],
            "measured_support_sizes":sorted(records),"declared_support_sizes":spec['support_sizes'],
            "reference_report":str(ROOT/spec['reference_report']),"reference_sha256":spec['reference_report_sha256'],
            "queries":256,"episodes":64,"seeds":seeds,"means":means,"per_seed":per_seed,"paired":contrasts,
            "harm_tolerance":harm_tolerance,"support_episode_cap":4,"support_start_gap":20,
            "schedule":spec.get('schedule','one shared initial update per CEM call'),
            "intervals":"10000 paired episode-bootstrap draws after seed averaging; pointwise descriptive 95%; seed20261110",
            "interpretation":"Support-budget sensitivity, fixed LR/update counts. Different B changes total training presentations and cost. P3 scores are model proxies, not physical success."}
    (directory/'overview.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,rows in [('means.csv',means),('per-seed.csv',per_seed)]:
        with (directory/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Fresh validation support-size ablation','',result['interpretation'],'',
          '| B | Setting | H5 MSE | Harm | Mean ESS | Seconds |','|---|---|---:|---:|---:|---:|']
    for row in means:
        ess='—' if row['effective_sample_size'] is None else f"{row['effective_sample_size']:.2f}"
        text.append(f"| {row['B']} | {row['setting']} | {row['H5']:.7f} | {row['harm_fraction']:.2%} | {ess} | {row['seconds']:.3f} |")
    text += ['', 'B32 values come from the pinned original report; they are not recomputed or counted as new experiments.',
             'Latencies span separately executed runs, so timing differences are descriptive rather than matched-wall-clock estimates.',
             'Paired H5 contrasts and pointwise episode-bootstrap intervals are in overview.json. No locks or test evaluations are created.']
    (directory/'overview.md').write_text('\n'.join(text)+'\n')
    print(f"Saved overview and per-seed CSV: {directory}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--support-sizes',type=int,nargs='+',help='Execute a subset of the declared grid')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--cpu-threads',type=int,default=8)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    require(args.cpu_threads>0,'cpu-threads must be positive')
    spec=read(args.config)
    require(spec['stage'] in ('p2','p3'),'Support-size pilot supports P2/P3')
    selected=spec['support_sizes'] if args.support_sizes is None else args.support_sizes
    require(selected and len(selected)==len(set(selected)) and set(selected)<=set(spec['support_sizes']),
            'Choose a distinct subset of the declared sizes')
    directory=(args.output_dir or ROOT/spec['experiment_dir']).resolve()
    source=reference(spec)
    prefix=executor(source,'p2_followup' if spec['stage']=='p2' else 'p3c')
    control=collect(source,source,spec,spec['reference_size'],control=True)
    plan={"specification":spec,"reference_code_fingerprint":identity(source['code']),"executor":prefix,
          "cpu_threads":args.cpu_threads,"runner_sha256":sha(Path(__file__)),
          "saved_helpers_sha256":sha(Path(__file__).with_name('saved_runs.py'))}
    plan_path=directory/'support-size-plan.json'
    if plan_path.exists():require(read(plan_path)==plan,'Existing plan differs; choose another --output-dir')
    jobs=[]
    for size in spec['support_sizes']:
        if size==spec['reference_size']:continue
        pp,p1,cp,c,expected=configurations(spec,source,size,directory)
        report_path=directory/'runs'/f'support-b{size}-validation'/'report.json'
        done=completed(report_path,source,spec,size,expected)
        jobs.append((size,pp,p1,cp,c,expected,report_path,done))
    print(f"Fresh validation only: {spec['stage']}, B={selected}, LR={spec['learning_rate']}, seeds={spec['seeds']}",flush=True)
    print(f"Reuse measured B32: {ROOT/spec['reference_report']}",flush=True)
    print('Reuse original observed latents, normalization and exact cuVS keys; no P1 preparation/re-encoding.',flush=True)
    print('Numerical executor:',shlex.join(prefix),flush=True)
    def run(command):
        print('RUN',shlex.join(command),flush=True)
        if not args.dry_run:subprocess.run(command,cwd=ROOT,check=True)
    pending=[j for j in jobs if j[0] in selected and j[-1] is None]
    if not args.dry_run and not args.analyze_only:immutable_json(plan_path,plan)
    if pending and not args.analyze_only:
        for _,pp,p1,cp,c,*_ in pending:
            if not args.dry_run:immutable_json(pp,p1);immutable_json(cp,c)
        run([*prefix,'prepare','--config',str(pending[0][3])])
    records={spec['reference_size']:control}
    for size,pp,p1,cp,c,expected,path,done in jobs:
        if done is not None:
            print('Skip verified completed:',path,flush=True)
        elif size in selected and not args.analyze_only:
            run([*prefix,'evaluate','--config',str(cp),'--split','validation','--run-name',path.parent.name,
                 '--seeds',*map(str,spec['seeds']),'--device','cuda','--cpu-threads',str(args.cpu_threads)])
            if not args.dry_run:
                done=completed(path,source,spec,size,expected)
                require(done is not None,'Evaluation did not complete')
        if done is not None:records[size]=collect(done,source,spec,size,expected=expected)
    if args.dry_run:
        print('Dry run complete: no model calls or metadata/output writes.',flush=True)
    else:analyze(spec,source,records,directory)


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,subprocess.CalledProcessError) as error:
        print(f'Support-size ablation failed: {error}',file=sys.stderr);raise SystemExit(1)
