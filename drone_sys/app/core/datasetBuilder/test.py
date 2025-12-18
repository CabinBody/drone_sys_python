import bluesky as bs
bs.init(guimode=False)
sim = bs.sim
print([m for m in dir(sim) if not m.startswith("_")])
