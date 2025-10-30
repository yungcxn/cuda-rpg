#ifndef INFOUNFOLD_H
#define INFOUNFOLD_H
/* X_TILEINFO_TUPLE(tileinfo_id, tx, ty) */
/* X_TILEINFO_TUPLE_ANIM(tileinfo_id, tx, ty, framelen_in_seconds) */

#define X_TILEINFO_COL(N, TILE_MACRO, id, row, tx, ty, ...) \
    X_TILEINFO_COL_##N(TILE_MACRO, id, row, tx, ty, ##__VA_ARGS__)

#define X_TILEINFO_COL_1(TILE_MACRO, id, row, tx, ty, ...) \
        TILE_MACRO(id##_0_##row, tx+0, ty+row, ##__VA_ARGS__)
#define X_TILEINFO_COL_2(TILE_MACRO, id, row, tx, ty, ...) \
        X_TILEINFO_COL_1(TILE_MACRO, id, row, tx, ty, ##__VA_ARGS__) TILE_MACRO(id##_1_##row, tx+1, ty+row, ##__VA_ARGS__)
#define X_TILEINFO_COL_3(TILE_MACRO, id, row, tx, ty, ...) \
        X_TILEINFO_COL_2(TILE_MACRO, id, row, tx, ty, ##__VA_ARGS__) TILE_MACRO(id##_2_##row, tx+2, ty+row, ##__VA_ARGS__)
#define X_TILEINFO_COL_4(TILE_MACRO, id, row, tx, ty, ...) \
        X_TILEINFO_COL_3(TILE_MACRO, id, row, tx, ty, ##__VA_ARGS__) TILE_MACRO(id##_3_##row, tx+3, ty+row, ##__VA_ARGS__)
#define X_TILEINFO_COL_5(TILE_MACRO, id, row, tx, ty, ...) \
        X_TILEINFO_COL_4(TILE_MACRO, id, row, tx, ty, ##__VA_ARGS__) TILE_MACRO(id##_4_##row, tx+4, ty+row, ##__VA_ARGS__)
#define X_TILEINFO_COL_6(TILE_MACRO, id, row, tx, ty, ...) \
        X_TILEINFO_COL_5(TILE_MACRO, id, row, tx, ty, ##__VA_ARGS__) TILE_MACRO(id##_5_##row, tx+5, ty+row, ##__VA_ARGS__)
#define X_TILEINFO_COL_7(TILE_MACRO, id, row, tx, ty, ...) \
        X_TILEINFO_COL_6(TILE_MACRO, id, row, tx, ty, ##__VA_ARGS__) TILE_MACRO(id##_6_##row, tx+6, ty+row, ##__VA_ARGS__)
#define X_TILEINFO_COL_8(TILE_MACRO, id, row, tx, ty, ...) \
        X_TILEINFO_COL_7(TILE_MACRO, id, row, tx, ty, ##__VA_ARGS__) TILE_MACRO(id##_7_##row, tx+7, ty+row, ##__VA_ARGS__)
#define X_TILEINFO_COL_9(TILE_MACRO, id, row, tx, ty, ...) \
        X_TILEINFO_COL_8(TILE_MACRO, id, row, tx, ty, ##__VA_ARGS__) TILE_MACRO(id##_8_##row, tx+8, ty+row, ##__VA_ARGS__)
#define X_TILEINFO_COL_10(TILE_MACRO, id, row, tx, ty, ...) \
        X_TILEINFO_COL_9(TILE_MACRO, id, row, tx, ty, ##__VA_ARGS__) TILE_MACRO(id##_9_##row, tx+9, ty+row, ##__VA_ARGS__)

#define X_TILEINFO_ROW(N, TILE_MACRO, id, tx, ty, A, ...) \
    X_TILEINFO_ROW_##N(TILE_MACRO, id, tx, ty, A, ##__VA_ARGS__)

#define X_TILEINFO_ROW_1(TILE_MACRO, id, tx, ty, A, ...) \
        X_TILEINFO_COL(A, TILE_MACRO, id, 0, tx, ty, ##__VA_ARGS__)
#define X_TILEINFO_ROW_2(TILE_MACRO, id, tx, ty, A, ...) \
        X_TILEINFO_ROW_1(TILE_MACRO, id, tx, ty, A, ##__VA_ARGS__) X_TILEINFO_COL(A, TILE_MACRO, id, 1, tx, ty, ##__VA_ARGS__)
#define X_TILEINFO_ROW_3(TILE_MACRO, id, tx, ty, A, ...) \
        X_TILEINFO_ROW_2(TILE_MACRO, id, tx, ty, A, ##__VA_ARGS__) X_TILEINFO_COL(A, TILE_MACRO, id, 2, tx, ty, ##__VA_ARGS__)
#define X_TILEINFO_ROW_4(TILE_MACRO, id, tx, ty, A, ...) \
        X_TILEINFO_ROW_3(TILE_MACRO, id, tx, ty, A, ##__VA_ARGS__) X_TILEINFO_COL(A, TILE_MACRO, id, 3, tx, ty, ##__VA_ARGS__)
#define X_TILEINFO_ROW_5(TILE_MACRO, id, tx, ty, A, ...) \
        X_TILEINFO_ROW_4(TILE_MACRO, id, tx, ty, A, ##__VA_ARGS__) X_TILEINFO_COL(A, TILE_MACRO, id, 4, tx, ty, ##__VA_ARGS__)
#define X_TILEINFO_ROW_6(TILE_MACRO, id, tx, ty, A, ...) \
        X_TILEINFO_ROW_5(TILE_MACRO, id, tx, ty, A, ##__VA_ARGS__) X_TILEINFO_COL(A, TILE_MACRO, id, 5, tx, ty, ##__VA_ARGS__)
#define X_TILEINFO_ROW_7(TILE_MACRO, id, tx, ty, A, ...) \
        X_TILEINFO_ROW_6(TILE_MACRO, id, tx, ty, A, ##__VA_ARGS__) X_TILEINFO_COL(A, TILE_MACRO, id, 6, tx, ty, ##__VA_ARGS__)
#define X_TILEINFO_ROW_8(TILE_MACRO, id, tx, ty, A, ...) \
        X_TILEINFO_ROW_7(TILE_MACRO, id, tx, ty, A, ##__VA_ARGS__) X_TILEINFO_COL(A, TILE_MACRO, id, 7, tx, ty, ##__VA_ARGS__)
#define X_TILEINFO_ROW_9(TILE_MACRO, id, tx, ty, A, ...) \
        X_TILEINFO_ROW_8(TILE_MACRO, id, tx, ty, A, ##__VA_ARGS__) X_TILEINFO_COL(A, TILE_MACRO, id, 8, tx, ty, ##__VA_ARGS__)
#define X_TILEINFO_ROW_10(TILE_MACRO, id, tx, ty, A, ...) \
        X_TILEINFO_ROW_9(TILE_MACRO, id, tx, ty, A, ##__VA_ARGS__) X_TILEINFO_COL(A, TILE_MACRO, id, 9, tx, ty, ##__VA_ARGS__)

#define X_TILEINFO_GRIDTUPLE(id, tx, ty, A, B) \
    X_TILEINFO_ROW(B, X_TILEINFO_TUPLE, id, tx, ty, A)

#define X_TILEINFO_GRIDTUPLE_ANIM(id, tx, ty, A, B, frame) \
    X_TILEINFO_ROW(B, X_TILEINFO_TUPLE_ANIM, id, tx, ty, A, frame)

#endif