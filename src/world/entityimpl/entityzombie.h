#ifndef ENTITYZOMBIE_H
#define ENTITYZOMBIE_H

#include <stdint.h>
#include "../../headeronly/def.h"

#ifdef __cplusplus
extern "C" {
#endif

void entityzombie_update(uint32_t entity_id, float64_t dt);

#ifdef __cplusplus
}
#endif

#endif