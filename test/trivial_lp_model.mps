* Min  2x - y
* s.t. x + y <= 3
*      0 <= x <= 1
*      1 <= y <= 2
NAME trivial_lp_model
ROWS
 N  OBJ
 L  con
COLUMNS
     x        con      1
     x        OBJ      2
     y        con      1
     y        OBJ      -1
RHS
    rhs       con      3
RANGES
BOUNDS
 LO bounds    y        1
 UP bounds    y        2
 LO bounds    x        0
 UP bounds    x        1
ENDATA
