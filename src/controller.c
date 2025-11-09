#include "controller.h"
#include "./headeronly/def.h"
#include "key.h"
#include "./world/bitfsm.h"

void controller_apply(world_ctx_t* world_ctx, key_inputfield_t pressed_keys,
                      key_inputfield_t released_keys) {
        if (!world_ctx || !world_ctx->player) return;

        player_t* player = world_ctx->player;
        bitfsm_state_t* playerstate = &(player->state);

        if (KEY_ON(pressed_keys, KEY_INPUT_W)) {
                bitfsm_tryexec_cb(playerstate, TOK(WALK_UP), player_bitfsm_callback);
        } else if (KEY_ON(pressed_keys, KEY_INPUT_S)) {
                bitfsm_tryexec_cb(playerstate, TOK(WALK_DOWN), player_bitfsm_callback);
        }

        if (KEY_ON(pressed_keys, KEY_INPUT_A)) {
                bitfsm_tryexec_cb(playerstate, TOK(WALK_LEFT), player_bitfsm_callback);
        } else if (KEY_ON(pressed_keys, KEY_INPUT_D)) {
                bitfsm_tryexec_cb(playerstate, TOK(WALK_RIGHT), player_bitfsm_callback);
        }

        if (KEY_ON(released_keys, KEY_INPUT_W)) {
                bitfsm_tryexec_cb(playerstate, TOK(END_WALK_UP), player_bitfsm_callback);
        } else if (KEY_ON(released_keys, KEY_INPUT_S)) {
                bitfsm_tryexec_cb(playerstate, TOK(END_WALK_DOWN), player_bitfsm_callback);
        }

        if (KEY_ON(released_keys, KEY_INPUT_A)) {
                bitfsm_tryexec_cb(playerstate, TOK(END_WALK_LEFT), player_bitfsm_callback);
        } else if (KEY_ON(released_keys, KEY_INPUT_D)) {
                bitfsm_tryexec_cb(playerstate, TOK(END_WALK_RIGHT), player_bitfsm_callback);
        }

        if (KEY_ON(pressed_keys, KEY_INPUT_SPACE)) {
                bitfsm_tryexec_cb(playerstate, TOK(ROLL), player_bitfsm_callback);
        } else if (KEY_ON(released_keys, KEY_INPUT_SPACE)) {
                bitfsm_tryexec_cb(playerstate, TOK(END_ROLL), player_bitfsm_callback);
        }

}