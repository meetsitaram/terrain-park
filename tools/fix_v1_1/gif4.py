import sys, os
sys.path.insert(0, 'tools/mirror'); import mirror_sets as ms; mc = ms.mc
takes, ctx = ms.load_all()
F = {'tp_stairs_113941': 'stairs1_side_approach_then_up_1_113941.gif', 'tp_driveway_121023': 'driveway_straight_uphill_1_121023.gif',
     'tp_driveway_121035': 'driveway_straight_downhill_1_121035.gif', 'tp_driveway_121106': 'driveway_diagonal_across_the_slope_1_121106.gif'}
for t in takes:
    if t['id'] in F:
        info = mc.render_gif(t, ctx, mc.HERE / 'gifs' / F[t['id']], strips=str(mc.HERE / 'strips')); print(F[t['id']], t['category'], info, flush=True)
