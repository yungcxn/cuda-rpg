#ifndef DEBUGTERM_H
#define DEBUGTERM_H

#include <ncurses.h>
#include <stdarg.h>
#include <stdio.h>

#define DEBUGTERM_BUFSIZE 2048

static inline void debugterm_init() {
        initscr();
        curs_set(0);
        nodelay(stdscr, TRUE);
        noecho();
}

static inline void debugterm_print(const char *fmt, ...) {
        char buffer[DEBUGTERM_BUFSIZE];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);

        clear();
        mvprintw(0, 0, "%s", buffer);
        refresh();
}

static inline void debugterm_end() {
        endwin();
}

#endif