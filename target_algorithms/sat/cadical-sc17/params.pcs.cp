arena categorical {1, 2, 3} [3]
arenacompact categorical {true, false} [true]
arenasort categorical {0, 1} [1] 
binary categorical {true, false} [true]
check categorical {true, false} [false]
compact categorical {true, false} [true]
compactint integer [10, 100000] [1000] 
compactlim real [0.001, 1] [0.1]
compactmin integer [1, 10000] [100]
elim categorical {true, false} [true]
elimclslim integer [10, 100000] [1000]
eliminit integer [10, 100000] [1000]
elimint integer [100, 1000000] [10000]
elimocclim integer [1, 10000] [100]
elimroundsinit integer [1, 500] [5] log
elimrounds integer [1, 200] [2] log
emagluefast real [0.0003, 1] [0.03]
emaglueslow real [1e-7, 1e-3] [1e-5]
emajump real [1e-7, 1e-3] [1e-5]
emasize real [1e-7, 1e-3] [1e-5]
decompose categorical {true, false} [true]
decomposerounds integer [1, 100] [1]
hbr categorical {true, false} [true]
hbrsizelim integer [10000000, 100000000000] [1000000000]
keepglue integer [1, 300] [3]
keepsize integer [1, 300] [3]
leak categorical {true, false} [true]
minimize categorical {true, false} [true]
minimizedepth integer [10, 100000] [1000]
phase categorical {0, 1} [1]
posize integer [1, 400] [4]
prefetch categorical {true, false} [true]
probe categorical {true, false} [true]
probeinit integer [5, 50000] [500]
probeint integer [100, 1000000] [10000]
probereleff real [0.0002, 1] [0.02]
probemaxeff real [1e5, 1e9] [1e7]
probemineff real [1000, 10000000] [100000]
reduceinc integer [3, 30000] [300]
reduceinit integer [20, 200000] [2000]
rephase categorical {true, false} [true]
rephaseint integer [1000, 10000000] [100000]
restart categorical {true, false} [true]
restartint integer [1, 600] [6]
restartmargin real [0, 10] [1.1]
reusetrail categorical {true, false} [true]
simplify categorical {true, false} [true]
strengthen categorical {true, false} [true]
subsume categorical {true, false} [true]
subsumebinlim integer [100, 100000] [10000]
subsumeclslim integer [10, 100000] [1000]
subsumeinc integer [100, 1000000] [10000]
subsumeinit integer [100, 1000000] [10000]
subsumeocclim integer [1, 10000] [100]
transred categorical {true, false} [true]
transredreleff real [0.001, 1] [0.1]
transredmaxeff real [1e5, 1e9] [1e7]
transredmineff real [1000, 10000000] [100000]
vivify categorical {true, false} [true]
vivifyreleff real [0.0003, 1] [0.03]
vivifymaxeff real [1e5, 1e9] [1e7]
vivifymineff real [1000, 10000000] [100000]
