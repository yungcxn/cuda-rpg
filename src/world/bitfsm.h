#ifndef BITFSM_H
#define BITFSM_H

#include "../headeronly/def.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "../key.h"

#define BITFSM_ISSET(state, mask) (((state) & (mask)) == (mask))
#define BITFSM_ISNOTSET(state, negmask) (((state) & (negmask)) == 0)
#define BITFSM_ANY(state, mask) (((state) & (mask)) != 0)

#define _BITFSM_STATE_INGAME_DEAD   BIT64(63)

#define _BITFSM_STATE_LOOK_UP       BIT64(KEY_INPUT_W) /* 0 */
#define _BITFSM_STATE_LOOK_DOWN     BIT64(KEY_INPUT_S) /* 1 */
#define _BITFSM_STATE_LOOK_LEFT     BIT64(KEY_INPUT_A) /* 2 */
#define _BITFSM_STATE_LOOK_RIGHT    BIT64(KEY_INPUT_D) /* 3 */

#define _BITFSM_STATE_WALK          BIT64(4)
#define _BITFSM_STATE_ROLL          BIT64(5)
#define _BITFSM_STATE_MOVING        (_BITFSM_STATE_WALK | _BITFSM_STATE_ROLL)

#define _BITFSM_STATE_MOVABLE       BIT64(6)

#define TOK(name) _BITFSM_TOK_##name
#define ST(name) _BITFSM_STATE_##name

#define X_BITFSM_LIVING_SUBTABLE \
        X(TOK(WALK_UP), ST(MOVABLE), 0, ST(WALK) | ST(LOOK_UP), 0) \
        X(TOK(END_WALK_UP), ST(WALK), 0, 0, ST(LOOK_UP)) \
        X(TOK(WALK_DOWN), ST(MOVABLE), 0, ST(WALK) | ST(LOOK_DOWN), 0) \
        X(TOK(END_WALK_DOWN), ST(WALK), 0, 0, ST(LOOK_DOWN)) \
        X(TOK(WALK_LEFT), ST(MOVABLE), 0, ST(WALK) | ST(LOOK_LEFT), 0) \
        X(TOK(END_WALK_LEFT), ST(WALK), 0, 0, ST(LOOK_LEFT)) \
        X(TOK(WALK_RIGHT), ST(MOVABLE), 0, ST(WALK) | ST(LOOK_RIGHT), 0) \
        X(TOK(END_WALK_RIGHT), ST(WALK), 0, 0, ST(LOOK_RIGHT)) \
        X(TOK(STOP_WALKING), 0, ST(MOVING), 0, ST(WALK)) \
        X(TOK(ROLL), ST(MOVABLE), 0, ST(ROLL), ST(MOVABLE)) \
        X(TOK(END_ROLL), ST(ROLL), 0, ST(MOVABLE), ST(ROLL))


#define X_BITFSM_TABLE \
        X_BITFSM_LIVING_SUBTABLE \
        /* X(...) */

typedef uint64_t bitfsm_state_t;
typedef uint32_t bitfsm_token_t;

#define X(token, t_mask, t_negmask, t_set, t_clear) token,
typedef enum {
        X_BITFSM_TABLE
        BITFSM_TOKEN_COUNT
} bitfsm_tokenname_t;
#undef X

typedef struct {
        bitfsm_state_t mask;     /* to check            */
        bitfsm_state_t negmask;  /* to check (negated)  */
        bitfsm_state_t set;      /* to set on match     */
        bitfsm_state_t clear;    /* to clear on match   */
} bitfsm_transition_t;

typedef void (*bitfsm_callback_t)(void* context, bitfsm_token_t tok);

void bitfsm_tryexec(bitfsm_state_t* state, const bitfsm_token_t t);
void bitfsm_tryexec_cb(bitfsm_state_t* state, const bitfsm_token_t t, bitfsm_callback_t callback);

#ifdef __cplusplus
}
#endif

#endif