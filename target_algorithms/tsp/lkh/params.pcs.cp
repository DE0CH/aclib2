ASCENT_CANDIDATES integer [10,500] [50] log
BACKBONE_TRIALS integer [0,5] [0]
BACKTRACKING categorical {YES,NO} [NO]
EXTRA_CANDIDATES integer [0,20] [0]
EXTRA_CANDIDATE_SET_TYPE categorical {NEAREST-NEIGHBOR,QUADRANT} [QUADRANT]
GAIN23 categorical {YES,NO} [YES]
GAIN_CRITERION categorical {YES,NO} [YES]
INITIAL_STEP_SIZE integer [1,5] [1]
INITIAL_TOUR_ALGORITHM categorical {BORUVKA,GREEDY,MOORE,NEAREST-NEIGHBOR,QUICK-BORUVKA,SIERPINSKI,WALK} [WALK]

#YP: I've added an extra categorical parameter to select whether or not the special WALK kick strategy is used. Normally this is selected with a value of 0 for KICK_TYPE and KICKS.
KICK_WALK categorical {NO,YES} [YES]
KICK_TYPE integer [4,20][12] #YP: This didn't have an initial default value so I chose the middle of the specified range
KICKS integer [1,5] [1]
#YP: Condition on the new categorical parameter
KICK_TYPE | KICK_WALK in {NO}
KICKS | KICK_WALK in {NO}

MAX_CANDIDATES integer [3,20] [5]
MOVE_TYPE integer [2,20] [5]
PATCHING_A integer [1,5] [2]
PATCHING_C integer [1,5] [3]
POPULATION_SIZE integer [0,1000] [0]
RESTRICTED_SEARCH categorical {YES,NO} [YES]
SUBGRADIENT categorical {YES,NO} [YES]

#YP: I've also added an extra categorical parameter to select whether or not the subsequent move types should be the same as the first move, since 0 normally encoded this special case
SAME_MOVE_TYPE categorical {YES,NO} [YES]
#YP: The value assigned to this parameter when LKH is run without it being specified is 5. Depite that the documentation indicates that the default value is 0.
SUBSEQUENT_MOVE_TYPE integer [2,20] [5]
SUBSEQUENT_MOVE_TYPE | SAME_MOVE_TYPE in {NO}

SUBSEQUENT_PATCHING categorical {YES,NO} [YES]

#YP: [PURE] is an optional suffix for DELAUNAY. I'm following zongxu by not using it (it seems to not work, though this may just be because I was specifying it incorrectly, I'm not really sure.
#CANDIDATE_SET_TYPE categorical {ALPHA,DELAUNAY[PURE],NEAREST-NEIGHBOR,QUADRANT} [ALPHA]
CANDIDATE_SET_TYPE categorical {ALPHA,DELAUNAY,NEAREST-NEIGHBOR,QUADRANT} [ALPHA]

