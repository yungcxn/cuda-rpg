#include "bitfsm.h"

#define X(token, t_mask, t_negmask, t_set, t_clear) \
        [token] = { t_mask, t_negmask, t_set, t_clear },
static const bitfsm_transition_t bitfsm_transitiontable[] = {
        X_BITFSM_TABLE
};
#undef X

void bitfsm_tryexec(bitfsm_state_t* state, const bitfsm_token_t t) {
        bitfsm_transition_t* transition = &bitfsm_transitiontable[t];
        if ((*state & transition->mask) == transition->mask && (*state & transition->negmask) == 0){
                *state |= transition->set;
                *state &= ~(transition->clear);
        }
}

void bitfsm_tryexec_cb(bitfsm_state_t* state, const bitfsm_token_t t, bitfsm_callback_t callback) {
        bitfsm_transition_t* transition = &bitfsm_transitiontable[t];
        if ((*state & transition->mask) == transition->mask && (*state & transition->negmask) == 0){
                *state |= transition->set;
                *state &= ~(transition->clear);
                if (callback) callback((void*)state, t);
        }
}