"""Task adapters are the only modules that interpret simulator-specific states."""


def environment_class(backend):
    if backend == "pusht":
        from .pusht import PushTEnvironment
        return PushTEnvironment
    raise ValueError(f"No simulator adapter registered for {backend!r}")


def make_environment(settings, dataset):
    if settings.get("task") == "green-target":
        from .green_pusht import GreenTargetPushT
        return GreenTargetPushT(settings, dataset)
    if getattr(dataset, "fresh_manifest", None) is not None:
        if settings["backend"] != "pusht":
            raise ValueError("Closed-loop replay adapters for this task are not implemented; fresh collection and offline P2/P3 are supported")
        from .fresh_recorded_pusht import FreshRecordedPushT
        return FreshRecordedPushT(settings, dataset)
    return environment_class(settings["backend"])(settings, dataset)


def fresh_environment_class(backend):
    """Collection task registry; generic storage/loops never interpret state."""
    if backend == "pusht":
        from .fresh_pusht import FreshPushT
        return FreshPushT
    if backend == "tworooms":
        from .fresh_tworooms import FreshTwoRooms
        return FreshTwoRooms
    if backend == "reacher":
        from .fresh_reacher import FreshReacher
        return FreshReacher
    if backend == "cube":
        from .fresh_cube import FreshCube
        return FreshCube
    raise ValueError(f"No fresh collection adapter registered for {backend!r}")


def make_collection_environment(config, regime):
    """Pass task-validated rendering and behavior settings through one factory."""
    adapter = fresh_environment_class(config["backend"])
    settings = {"rendering": config["rendering"]} if "rendering" in config else {}
    if hasattr(adapter, "collection_options"):
        settings.update(adapter.collection_options(config))
    return adapter(config["image_size"], config["regimes"][regime], **settings)
