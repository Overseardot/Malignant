// ============================================================================
//  SimpleEngine 1.0 — A UCI Chess Engine in C++17
//  --------------------------------------------------------------------------
//  Single-file implementation. No external dependencies.
//
//  Build:
//      g++ -O2 -std=c++17 -o engine engine.cpp
//      (or clang++ / MSVC cl /O2 engine.cpp)
//
//  Quick Makefile (optional):
//      CXX      = g++
//      CXXFLAGS = -O2 -std=c++17 -Wall
//      engine: engine.cpp
//          $(CXX) $(CXXFLAGS) -o $@ $<
//
//  Features:
//    - Bitboard move generation (ray attacks), mailbox + Zobrist incremental hash
//    - Alpha-Beta / PVS, iterative deepening, transposition table
//    - Null-move pruning, late-move reductions, check extension
//    - Killer moves + history heuristic, MVV-LVA move ordering
//    - Quiescence search with full evasion generation
//    - Tapered evaluation: material, PSTs, bishop pair, passed/doubled pawns
//    - UCI protocol: uci / isready / position / go / perft / setoption Hash
//
//  Author: <you>
//  License: MIT (or whatever you prefer)
// ============================================================================

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

typedef uint64_t U64;
typedef uint32_t Move;

// ------------------------------- constants ----------------------------------
enum { WHITE = 0, BLACK = 1 };
enum { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum { EMPTY = 12 };

const int MAX_PLY      = 128;
const int INF          = 1000000;
const int MATE         = 100000;
const int MATE_BOUND   = MATE - 1000;

const U64 FILE_A = 0x0101010101010101ULL;
const U64 FILE_H = 0x8080808080808080ULL;
const U64 RANK1  = 0x00000000000000FFULL;
const U64 RANK2  = 0x000000000000FF00ULL;
const U64 RANK3  = 0x0000000000FF0000ULL;
const U64 RANK6  = 0x0000FF0000000000ULL;
const U64 RANK7  = 0x00FF000000000000ULL;
const U64 RANK8  = 0xFF00000000000000ULL;

// ----------------------------- move encoding --------------------------------
// bits 0-5  : from
// bits 6-11 : to
// bits 12-14: promotion (0 none, 1=N, 2=B, 3=R, 4=Q)
// bit 15    : capture
// bit 16    : en-passant
// bit 17    : castle
// bit 18    : double pawn push
#define FROM(m)       ((m) & 63)
#define TO(m)         (((m) >> 6) & 63)
#define PROMO(m)      (((m) >> 12) & 7)
#define IS_CAP(m)     ((m) & (1u << 15))
#define IS_EP(m)      ((m) & (1u << 16))
#define IS_CASTLE(m)  ((m) & (1u << 17))
#define IS_DBL(m)     ((m) & (1u << 18))

enum { FLAG_CAP = 1, FLAG_EP = 2, FLAG_CASTLE = 4, FLAG_DBL = 8 };

#define MK_MOVE(f, t, pr, fl) \
    ((Move)((f) | ((t) << 6) | ((pr) << 12) | ((fl) << 15)))

inline int poplsb(U64& b) { int s = __builtin_ctzll(b); b &= b - 1; return s; }
inline int lsb(U64 b)     { return __builtin_ctzll(b); }
inline int popcount(U64 b) { return __builtin_popcountll(b); }

// ------------------------------ attack tables -------------------------------
U64 knightAtt[64], kingAtt[64], pawnAtt[2][64];
U64 frontSpanW[64], frontSpanB[64];

static const int rookDirs[4][2]   = {{1,0},{-1,0},{0,1},{0,-1}};
static const int bishopDirs[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};

inline U64 rayAttacks(int sq, U64 occ, const int dirs[4][2]) {
    U64 att = 0;
    int r0 = sq >> 3, f0 = sq & 7;
    for (int d = 0; d < 4; d++) {
        int r = r0 + dirs[d][0], f = f0 + dirs[d][1];
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            int s = r * 8 + f;
            att |= 1ULL << s;
            if (occ & (1ULL << s)) break;
            r += dirs[d][0]; f += dirs[d][1];
        }
    }
    return att;
}
inline U64 bishopAttacks(int sq, U64 occ) { return rayAttacks(sq, occ, bishopDirs); }
inline U64 rookAttacks  (int sq, U64 occ) { return rayAttacks(sq, occ, rookDirs);   }

// -------------------------------- zobrist -----------------------------------
struct Zobrist {
    U64 piece[12][64];
    U64 castling[16];
    U64 ep[64];
    U64 side;
} Z;

static U64 rngState = 0x1234567890ABCDEFULL;
static U64 rand64() {
    rngState ^= rngState >> 12;
    rngState ^= rngState << 25;
    rngState ^= rngState >> 27;
    return rngState * 2685821657736338717ULL;
}

// ---------------------------- evaluation tables -----------------------------
const int pieceValue[6] = { 100, 320, 330, 500, 900, 0 };

const int PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};
const int PST_KNIGHT[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};
const int PST_BISHOP[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};
const int PST_ROOK[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
};
const int PST_QUEEN[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
};
const int PST_KING_MG[64] = {
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
};
const int PST_KING_EG[64] = {
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50
};
const int passedBonus[8] = { 0, 5, 10, 20, 35, 60, 100, 0 };

inline const int* pstFor(int type) {
    switch (type) {
        case PAWN:   return PST_PAWN;
        case KNIGHT: return PST_KNIGHT;
        case BISHOP: return PST_BISHOP;
        case ROOK:   return PST_ROOK;
        case QUEEN:  return PST_QUEEN;
        default:     return PST_QUEEN;
    }
}

// --------------------------------- position ---------------------------------
struct Position {
    U64 byType[6];
    U64 byColor[2];
    U64 occ;
    int8_t board[64];
    int side;
    int ep;          // -1 or square
    int castling;    // 1=WK 2=WQ 4=BK 8=BQ
    int halfmove;
    int fullmove;
    U64 hash;
    U64 hist[4096];
    int histLen;
};

Position P;

inline void putPiece(Position& p, int sq, int pc) {
    U64 b = 1ULL << sq;
    int c = pc / 6, t = pc % 6;
    p.byType[t] |= b; p.byColor[c] |= b; p.occ |= b;
    p.board[sq] = (int8_t)pc;
    p.hash ^= Z.piece[pc][sq];
}
inline void removePiece(Position& p, int sq) {
    int pc = p.board[sq];
    U64 b = 1ULL << sq;
    int c = pc / 6, t = pc % 6;
    p.byType[t] &= ~b; p.byColor[c] &= ~b; p.occ &= ~b;
    p.board[sq] = EMPTY;
    p.hash ^= Z.piece[pc][sq];
}
inline int kingSq(const Position& p, int c) {
    return lsb(p.byType[KING] & p.byColor[c]);
}

struct Undo {
    int captured;
    int castling;
    int ep;
    int halfmove;
    U64 hash;
};

struct MoveList {
    Move m[256];
    int  score[256];
    int  n;
    MoveList() : n(0) {}
    void add(Move mv) { m[n] = mv; score[n] = 0; n++; }
};

// -------------------------------- init --------------------------------------
void init() {
    for (int sq = 0; sq < 64; sq++) {
        int r = sq >> 3, f = sq & 7;
        U64 n = 0, k = 0, pw = 0, pb = 0;
        static const int kn[8][2] = {{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
        for (auto& d : kn) {
            int rr = r + d[0], ff = f + d[1];
            if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) n |= 1ULL << (rr*8+ff);
        }
        for (int dr = -1; dr <= 1; dr++)
            for (int df = -1; df <= 1; df++) {
                if (!dr && !df) continue;
                int rr = r + dr, ff = f + df;
                if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) k |= 1ULL << (rr*8+ff);
            }
        if (r < 7 && f < 7) pw |= 1ULL << ((r+1)*8 + (f+1));
        if (r < 7 && f > 0) pw |= 1ULL << ((r+1)*8 + (f-1));
        if (r > 0 && f < 7) pb |= 1ULL << ((r-1)*8 + (f+1));
        if (r > 0 && f > 0) pb |= 1ULL << ((r-1)*8 + (f-1));
        knightAtt[sq] = n;
        kingAtt[sq]   = k;
        pawnAtt[WHITE][sq] = pw;
        pawnAtt[BLACK][sq] = pb;

        U64 ws = 0, bs = 0;
        for (int rr = r + 1; rr < 8; rr++)
            for (int ff = std::max(0, f-1); ff <= std::min(7, f+1); ff++)
                ws |= 1ULL << (rr*8+ff);
        for (int rr = r - 1; rr >= 0; rr--)
            for (int ff = std::max(0, f-1); ff <= std::min(7, f+1); ff++)
                bs |= 1ULL << (rr*8+ff);
        frontSpanW[sq] = ws;
        frontSpanB[sq] = bs;
    }

    rngState = 0x1234567890ABCDEFULL;
    for (int pc = 0; pc < 12; pc++)
        for (int sq = 0; sq < 64; sq++)
            Z.piece[pc][sq] = rand64();
    for (int i = 0; i < 16; i++) Z.castling[i] = rand64();
    for (int i = 0; i < 64; i++) Z.ep[i] = rand64();
    Z.side = rand64();
}

// --------------------------------- FEN --------------------------------------
void parseFen(Position& p, const std::string& fen) {
    memset(p.byType, 0, sizeof(p.byType));
    memset(p.byColor, 0, sizeof(p.byColor));
    p.occ = 0; p.hash = 0; p.ep = -1; p.castling = 0;
    p.halfmove = 0; p.fullmove = 1; p.histLen = 0;
    for (int i = 0; i < 64; i++) p.board[i] = EMPTY;

    std::istringstream ss(fen);
    std::string boardStr, sideStr, castleStr, epStr;
    ss >> boardStr >> sideStr >> castleStr >> epStr;

    int rank = 7, file = 0;
    for (char c : boardStr) {
        if (c == '/') { rank--; file = 0; }
        else if (c >= '1' && c <= '8') file += c - '0';
        else {
            int color = (c >= 'A' && c <= 'Z') ? WHITE : BLACK;
            char lc = (char)tolower(c);
            int type = (lc=='p')?PAWN:(lc=='n')?KNIGHT:(lc=='b')?BISHOP:
                       (lc=='r')?ROOK:(lc=='q')?QUEEN:KING;
            putPiece(p, rank*8 + file, color*6 + type);
            file++;
        }
    }
    p.side = (sideStr == "b") ? BLACK : WHITE;
    if (p.side == BLACK) p.hash ^= Z.side;

    if (castleStr.find('K') != std::string::npos) p.castling |= 1;
    if (castleStr.find('Q') != std::string::npos) p.castling |= 2;
    if (castleStr.find('k') != std::string::npos) p.castling |= 4;
    if (castleStr.find('q') != std::string::npos) p.castling |= 8;
    p.hash ^= Z.castling[p.castling];

    if (epStr.size() == 2 && epStr[0] != '-') {
        int f = epStr[0] - 'a', r = epStr[1] - '1';
        p.ep = r*8 + f;
        p.hash ^= Z.ep[p.ep];
    }
    if (!(ss >> p.halfmove)) p.halfmove = 0;
    if (!(ss >> p.fullmove)) p.fullmove = 1;
}

void setStartPos(Position& p) {
    parseFen(p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

// ------------------------------ attack test ---------------------------------
bool isAttacked(const Position& p, int sq, int by) {
    if (pawnAtt[by ^ 1][sq] & p.byType[PAWN] & p.byColor[by]) return true;
    if (knightAtt[sq] & p.byType[KNIGHT] & p.byColor[by])     return true;
    if (kingAtt[sq]   & p.byType[KING]   & p.byColor[by])     return true;
    U64 bq = (p.byType[BISHOP] | p.byType[QUEEN]) & p.byColor[by];
    if (bq && (bishopAttacks(sq, p.occ) & bq)) return true;
    U64 rq = (p.byType[ROOK] | p.byType[QUEEN]) & p.byColor[by];
    if (rq && (rookAttacks(sq, p.occ) & rq))   return true;
    return false;
}

// ---------------------------- move generation -------------------------------
void genMoves(const Position& p, MoveList& ml, bool capsOnly) {
    int us = p.side, them = us ^ 1;
    U64 own = p.byColor[us], enemy = p.byColor[them];
    U64 occ = p.occ, empty = ~occ;

    // ---------------- pawns ----------------
    U64 pawns = p.byType[PAWN] & own;
    if (us == WHITE) {
        U64 single = (pawns << 8) & empty;
        U64 dbl    = ((single & RANK3) << 8) & empty;
        U64 promoP = single & RANK8;
        U64 quietP = single & ~RANK8;
        if (!capsOnly) {
            while (quietP) { int t = poplsb(quietP); ml.add(MK_MOVE(t-8,  t, 0, 0)); }
            while (dbl)    { int t = poplsb(dbl);    ml.add(MK_MOVE(t-16, t, 0, FLAG_DBL)); }
        }
        while (promoP) {
            int t = poplsb(promoP);
            for (int pr = 4; pr >= 1; pr--) ml.add(MK_MOVE(t-8, t, pr, 0));
        }
        U64 capL = ((pawns & ~FILE_A) << 7) & enemy;
        U64 capR = ((pawns & ~FILE_H) << 9) & enemy;
        U64 pcL = capL & RANK8, ncL = capL & ~RANK8;
        U64 pcR = capR & RANK8, ncR = capR & ~RANK8;
        while (ncL) { int t = poplsb(ncL); ml.add(MK_MOVE(t-7, t, 0, FLAG_CAP)); }
        while (ncR) { int t = poplsb(ncR); ml.add(MK_MOVE(t-9, t, 0, FLAG_CAP)); }
        while (pcL) { int t = poplsb(pcL); for (int pr = 4; pr >= 1; pr--) ml.add(MK_MOVE(t-7, t, pr, FLAG_CAP)); }
        while (pcR) { int t = poplsb(pcR); for (int pr = 4; pr >= 1; pr--) ml.add(MK_MOVE(t-9, t, pr, FLAG_CAP)); }
    } else {
        U64 single = (pawns >> 8) & empty;
        U64 dbl    = ((single & RANK6) >> 8) & empty;
        U64 promoP = single & RANK1;
        U64 quietP = single & ~RANK1;
        if (!capsOnly) {
            while (quietP) { int t = poplsb(quietP); ml.add(MK_MOVE(t+8,  t, 0, 0)); }
            while (dbl)    { int t = poplsb(dbl);    ml.add(MK_MOVE(t+16, t, 0, FLAG_DBL)); }
        }
        while (promoP) {
            int t = poplsb(promoP);
            for (int pr = 4; pr >= 1; pr--) ml.add(MK_MOVE(t+8, t, pr, 0));
        }
        U64 capL = ((pawns & ~FILE_A) >> 9) & enemy;
        U64 capR = ((pawns & ~FILE_H) >> 7) & enemy;
        U64 pcL = capL & RANK1, ncL = capL & ~RANK1;
        U64 pcR = capR & RANK1, ncR = capR & ~RANK1;
        while (ncL) { int t = poplsb(ncL); ml.add(MK_MOVE(t+9, t, 0, FLAG_CAP)); }
        while (ncR) { int t = poplsb(ncR); ml.add(MK_MOVE(t+7, t, 0, FLAG_CAP)); }
        while (pcL) { int t = poplsb(pcL); for (int pr = 4; pr >= 1; pr--) ml.add(MK_MOVE(t+9, t, pr, FLAG_CAP)); }
        while (pcR) { int t = poplsb(pcR); for (int pr = 4; pr >= 1; pr--) ml.add(MK_MOVE(t+7, t, pr, FLAG_CAP)); }
    }

    // ---------------- en passant ----------------
    if (p.ep >= 0) {
        U64 attackers = pawnAtt[them][p.ep] & pawns;
        while (attackers) {
            int s = poplsb(attackers);
            ml.add(MK_MOVE(s, p.ep, 0, FLAG_CAP | FLAG_EP));
        }
    }

    // ---------------- knights ----------------
    U64 bb = p.byType[KNIGHT] & own;
    while (bb) {
        int s = poplsb(bb);
        U64 att = knightAtt[s] & ~own;
        if (capsOnly) att &= enemy;
        while (att) {
            int t = poplsb(att);
            ml.add(MK_MOVE(s, t, 0, (enemy >> t & 1) ? FLAG_CAP : 0));
        }
    }

    // ---------------- bishops / rooks / queens ----------------
    bb = p.byType[BISHOP] & own;
    while (bb) {
        int s = poplsb(bb);
        U64 att = bishopAttacks(s, occ) & ~own;
        if (capsOnly) att &= enemy;
        while (att) { int t = poplsb(att); ml.add(MK_MOVE(s, t, 0, (enemy >> t & 1) ? FLAG_CAP : 0)); }
    }
    bb = p.byType[ROOK] & own;
    while (bb) {
        int s = poplsb(bb);
        U64 att = rookAttacks(s, occ) & ~own;
        if (capsOnly) att &= enemy;
        while (att) { int t = poplsb(att); ml.add(MK_MOVE(s, t, 0, (enemy >> t & 1) ? FLAG_CAP : 0)); }
    }
    bb = p.byType[QUEEN] & own;
    while (bb) {
        int s = poplsb(bb);
        U64 att = (bishopAttacks(s, occ) | rookAttacks(s, occ)) & ~own;
        if (capsOnly) att &= enemy;
        while (att) { int t = poplsb(att); ml.add(MK_MOVE(s, t, 0, (enemy >> t & 1) ? FLAG_CAP : 0)); }
    }

    // ---------------- king ----------------
    bb = p.byType[KING] & own;
    while (bb) {
        int s = poplsb(bb);
        U64 att = kingAtt[s] & ~own;
        if (capsOnly) att &= enemy;
        while (att) { int t = poplsb(att); ml.add(MK_MOVE(s, t, 0, (enemy >> t & 1) ? FLAG_CAP : 0)); }
    }

    // ---------------- castling ----------------
    if (!capsOnly) {
        if (us == WHITE) {
            if ((p.castling & 1) && !(occ & ((1ULL<<5)|(1ULL<<6)))
                && !isAttacked(p, 4, BLACK) && !isAttacked(p, 5, BLACK) && !isAttacked(p, 6, BLACK))
                ml.add(MK_MOVE(4, 6, 0, FLAG_CASTLE));
            if ((p.castling & 2) && !(occ & ((1ULL<<1)|(1ULL<<2)|(1ULL<<3)))
                && !isAttacked(p, 4, BLACK) && !isAttacked(p, 3, BLACK) && !isAttacked(p, 2, BLACK))
                ml.add(MK_MOVE(4, 2, 0, FLAG_CASTLE));
        } else {
            if ((p.castling & 4) && !(occ & ((1ULL<<61)|(1ULL<<62)))
                && !isAttacked(p, 60, WHITE) && !isAttacked(p, 61, WHITE) && !isAttacked(p, 62, WHITE))
                ml.add(MK_MOVE(60, 62, 0, FLAG_CASTLE));
            if ((p.castling & 8) && !(occ & ((1ULL<<57)|(1ULL<<58)|(1ULL<<59)))
                && !isAttacked(p, 60, WHITE) && !isAttacked(p, 59, WHITE) && !isAttacked(p, 58, WHITE))
                ml.add(MK_MOVE(60, 58, 0, FLAG_CASTLE));
        }
    }
}

// ------------------------------ make / unmake -------------------------------
inline void clearCastlingSquare(Position& p, int sq) {
    switch (sq) {
        case 0:  p.castling &= ~2; break;
        case 7:  p.castling &= ~1; break;
        case 56: p.castling &= ~8; break;
        case 63: p.castling &= ~4; break;
    }
}

void makeMove(Position& p, Move m, Undo& u) {
    u.captured = EMPTY;
    u.castling = p.castling;
    u.ep       = p.ep;
    u.halfmove = p.halfmove;
    u.hash     = p.hash;

    int from = FROM(m), to = TO(m);
    int us = p.side, them = us ^ 1;
    int pc = p.board[from];

    if (IS_EP(m)) {
        int capSq = to + (us == WHITE ? -8 : 8);
        u.captured = p.board[capSq];
        removePiece(p, capSq);
    } else if (p.board[to] != EMPTY) {
        u.captured = p.board[to];
        removePiece(p, to);
    }

    removePiece(p, from);
    if (PROMO(m)) putPiece(p, to, us*6 + (PROMO(m) - 1 + KNIGHT));
    else          putPiece(p, to, pc);

    if (IS_CASTLE(m)) {
        int rf, rt;
        switch (to) {
            case 6:  rf = 7;  rt = 5;  break;
            case 2:  rf = 0;  rt = 3;  break;
            case 62: rf = 63; rt = 61; break;
            default: rf = 56; rt = 59; break;
        }
        int rook = p.board[rf];
        removePiece(p, rf);
        putPiece(p, rt, rook);
    }

    if (p.ep >= 0) { p.hash ^= Z.ep[p.ep]; p.ep = -1; }
    if (IS_DBL(m)) {
        p.ep = (from + to) / 2;
        p.hash ^= Z.ep[p.ep];
    }

    p.hash ^= Z.castling[p.castling];
    if (pc == us*6 + KING) p.castling &= (us == WHITE) ? ~3 : ~12;
    clearCastlingSquare(p, from);
    clearCastlingSquare(p, to);
    p.hash ^= Z.castling[p.castling];

    if (pc == us*6 + PAWN || u.captured != EMPTY) p.halfmove = 0;
    else p.halfmove++;
    if (us == BLACK) p.fullmove++;

    p.side = them;
    p.hash ^= Z.side;

    if (p.histLen < 4096) p.hist[p.histLen++] = p.hash;
}

void unmakeMove(Position& p, Move m, const Undo& u) {
    p.side ^= 1;
    int us = p.side;
    int from = FROM(m), to = TO(m);

    if (PROMO(m)) {
        removePiece(p, to);
        putPiece(p, from, us*6 + PAWN);
    } else {
        int pc = p.board[to];
        removePiece(p, to);
        putPiece(p, from, pc);
    }

    if (IS_CASTLE(m)) {
        int rf, rt;
        switch (to) {
            case 6:  rf = 7;  rt = 5;  break;
            case 2:  rf = 0;  rt = 3;  break;
            case 62: rf = 63; rt = 61; break;
            default: rf = 56; rt = 59; break;
        }
        int rook = p.board[rt];
        removePiece(p, rt);
        putPiece(p, rf, rook);
    }

    if (IS_EP(m)) {
        int capSq = to + (us == WHITE ? -8 : 8);
        putPiece(p, capSq, (us^1)*6 + PAWN);
    } else if (u.captured != EMPTY) {
        putPiece(p, to, u.captured);
    }

    p.castling = u.castling;
    p.ep       = u.ep;
    p.halfmove = u.halfmove;
    p.hash     = u.hash;
    if (us == BLACK) p.fullmove--;
    p.histLen--;
}

void makeNull(Position& p, Undo& u) {
    u.castling = p.castling;
    u.ep       = p.ep;
    u.halfmove = p.halfmove;
    u.hash     = p.hash;
    u.captured = EMPTY;
    if (p.ep >= 0) { p.hash ^= Z.ep[p.ep]; p.ep = -1; }
    p.side ^= 1;
    p.hash ^= Z.side;
    p.halfmove++;
    if (p.histLen < 4096) p.hist[p.histLen++] = p.hash;
}
void unmakeNull(Position& p, const Undo& u) {
    p.side ^= 1;
    p.castling = u.castling;
    p.ep       = u.ep;
    p.halfmove = u.halfmove;
    p.hash     = u.hash;
    p.histLen--;
}

// ------------------------------- evaluation ---------------------------------
int evaluate(const Position& p) {
    int mg = 0, eg = 0;

    for (int c = 0; c < 2; c++) {
        int sign = (c == WHITE) ? 1 : -1;
        for (int t = 0; t < 6; t++) {
            U64 bb = p.byType[t] & p.byColor[c];
            while (bb) {
                int sq = poplsb(bb);
                int idx = (c == WHITE) ? (sq ^ 56) : sq;
                if (t == KING) {
                    mg += sign * PST_KING_MG[idx];
                    eg += sign * PST_KING_EG[idx];
                } else {
                    int v = pieceValue[t] + pstFor(t)[idx];
                    mg += sign * v;
                    eg += sign * v;
                }
            }
        }
    }

    int phase = 4 * popcount(p.byType[QUEEN])
              + 2 * popcount(p.byType[ROOK])
              +     popcount(p.byType[KNIGHT])
              +     popcount(p.byType[BISHOP]);
    if (phase > 24) phase = 24;

    int score = (mg * phase + eg * (24 - phase)) / 24;

    for (int c = 0; c < 2; c++) {
        if (popcount(p.byType[BISHOP] & p.byColor[c]) >= 2)
            score += (c == WHITE ? 30 : -30);
    }

    for (int c = 0; c < 2; c++) {
        U64 pawns = p.byType[PAWN] & p.byColor[c];
        U64 enemyPawns = p.byType[PAWN] & p.byColor[c ^ 1];
        int sign = (c == WHITE) ? 1 : -1;
        for (int f = 0; f < 8; f++) {
            U64 fileMask = 0x0101010101010101ULL << f;
            int cnt = popcount(pawns & fileMask);
            if (cnt > 1) score -= sign * 10 * (cnt - 1);
        }
        U64 bb = pawns;
        while (bb) {
            int sq = poplsb(bb);
            U64 front = (c == WHITE) ? frontSpanW[sq] : frontSpanB[sq];
            if (!(front & enemyPawns)) {
                int r = (c == WHITE) ? (sq >> 3) : (7 - (sq >> 3));
                score += sign * passedBonus[r];
            }
        }
    }

    return (p.side == WHITE) ? score : -score;
}

// ------------------------------ transposition -------------------------------
enum { TT_EXACT = 1, TT_ALPHA = 2, TT_BETA = 3 };

struct TTEntry {
    U64 key;
    int32_t score;
    Move    move;
    int16_t depth;
    uint8_t flag;
};

std::vector<TTEntry> TT;
U64 ttMask = 0;

void ttInit(int mb) {
    size_t entries = (size_t)mb * 1024 * 1024 / sizeof(TTEntry);
    size_t n = 1;
    while (n * 2 <= entries) n *= 2;
    TT.assign(n, TTEntry{});
    ttMask = n - 1;
}
void ttClear() { std::fill(TT.begin(), TT.end(), TTEntry{}); }

// -------------------------------- search ------------------------------------
Move killers[MAX_PLY][2];
int  historyTab[64][64];
Move pvTable[MAX_PLY][MAX_PLY];
int  pvLen[MAX_PLY];

int  nodes = 0;
bool stopped = false;
std::chrono::steady_clock::time_point startTime;
int  timeLimitMs = 0;
Move bestRootMove = 0;

int elapsedMs() {
    return (int)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count();
}
inline bool timeUp() {
    if (timeLimitMs <= 0) return false;
    return elapsedMs() >= timeLimitMs;
}

bool isRepetition() {
    int end = P.histLen - 1 - P.halfmove;
    if (end < 0) end = 0;
    for (int i = P.histLen - 3; i >= end; i -= 2)
        if (P.hist[i] == P.hash) return true;
    return false;
}

bool hasNonPawnMaterial(int c) {
    return (P.byType[KNIGHT] | P.byType[BISHOP] |
            P.byType[ROOK]   | P.byType[QUEEN]) & P.byColor[c];
}

void ttStore(U64 key, int depth, int score, Move move, int flag, int ply) {
    TTEntry& e = TT[key & ttMask];
    if (e.flag == 0 || e.key == key || depth >= e.depth) {
        if (score >= MATE_BOUND)       score += ply;
        else if (score <= -MATE_BOUND) score -= ply;
        e.key   = key;
        e.depth = (int16_t)depth;
        e.score = score;
        e.move  = move;
        e.flag  = (uint8_t)flag;
    }
}

void scoreMoves(MoveList& ml, Move ttMove, int ply) {
    for (int i = 0; i < ml.n; i++) {
        Move m = ml.m[i];
        int s;
        if (m == ttMove) s = 10000000;
        else if (IS_CAP(m) || IS_EP(m)) {
            int victim   = IS_EP(m) ? PAWN : (P.board[TO(m)] % 6);
            int attacker = P.board[FROM(m)] % 6;
            s = 1000000 + victim * 100 - attacker * 10;
        } else if (PROMO(m)) {
            s = 900000 + PROMO(m);
        } else if (m == killers[ply][0]) s = 800000;
        else if (m == killers[ply][1])   s = 790000;
        else s = historyTab[FROM(m)][TO(m)];
        ml.score[i] = s;
    }
}

int quiesce(int alpha, int beta, int ply) {
    if (stopped) return 0;
    pvLen[ply] = ply;

    nodes++;
    if ((nodes & 2047) == 0 && timeUp()) { stopped = true; return 0; }
    if (ply >= MAX_PLY - 2) return evaluate(P);

    bool inCheck = isAttacked(P, kingSq(P, P.side), P.side ^ 1);

    int bestScore;
    if (!inCheck) {
        bestScore = evaluate(P);
        if (bestScore >= beta) return bestScore;
        if (bestScore > alpha) alpha = bestScore;
    } else {
        bestScore = -INF;
    }

    MoveList ml;
    genMoves(P, ml, !inCheck);
    scoreMoves(ml, 0, ply);

    int legal = 0;
    for (int i = 0; i < ml.n; i++) {
        int bi = i;
        for (int j = i + 1; j < ml.n; j++)
            if (ml.score[j] > ml.score[bi]) bi = j;
        std::swap(ml.m[i], ml.m[bi]);
        std::swap(ml.score[i], ml.score[bi]);

        Move m = ml.m[i];
        Undo u;
        makeMove(P, m, u);
        if (isAttacked(P, kingSq(P, P.side ^ 1), P.side)) { unmakeMove(P, m, u); continue; }
        legal++;

        int score = -quiesce(-beta, -alpha, ply + 1);
        unmakeMove(P, m, u);
        if (stopped) return 0;

        if (score > bestScore) {
            bestScore = score;
            if (score > alpha) {
                alpha = score;
                pvTable[ply][ply] = m;
                for (int k = ply + 1; k < pvLen[ply + 1]; k++)
                    pvTable[ply][k] = pvTable[ply + 1][k];
                pvLen[ply] = pvLen[ply + 1];
                if (score >= beta) break;
            }
        }
    }
    if (inCheck && legal == 0) return -MATE + ply;
    return bestScore;
}

int alphaBeta(int depth, int alpha, int beta, int ply, bool canNull) {
    if (stopped) return 0;
    pvLen[ply] = ply;

    if (ply > 0) {
        if (isRepetition() || P.halfmove >= 100) return 0;
        if (alpha < -MATE + ply)      alpha = -MATE + ply;
        if (beta  >  MATE - ply - 1)  beta  =  MATE - ply - 1;
        if (alpha >= beta) return alpha;
    }

    bool inCheck = isAttacked(P, kingSq(P, P.side), P.side ^ 1);
    if (inCheck) depth++;
    if (depth <= 0) return quiesce(alpha, beta, ply);
    if (ply >= MAX_PLY - 2) return evaluate(P);

    nodes++;
    if ((nodes & 2047) == 0 && timeUp()) { stopped = true; return 0; }

    TTEntry* tte = &TT[P.hash & ttMask];
    Move ttMove = 0;
    if (tte->flag && tte->key == P.hash) {
        ttMove = tte->move;
        if (ply > 0 && tte->depth >= depth) {
            int s = tte->score;
            if (s >= MATE_BOUND)       s -= ply;
            else if (s <= -MATE_BOUND) s += ply;
            if (tte->flag == TT_EXACT) return s;
            if (tte->flag == TT_ALPHA && s <= alpha) return s;
            if (tte->flag == TT_BETA  && s >= beta)  return s;
        }
    }

    if (canNull && !inCheck && depth >= 3 && ply > 0 && hasNonPawnMaterial(P.side)) {
        Undo nu;
        makeNull(P, nu);
        int R = 2 + depth / 6;
        int s = -alphaBeta(depth - 1 - R, -beta, -beta + 1, ply + 1, false);
        unmakeNull(P, nu);
        if (stopped) return 0;
        if (s >= beta) return beta;
    }

    MoveList ml;
    genMoves(P, ml, false);
    scoreMoves(ml, ttMove, ply);

    int bestScore = -INF;
    Move bestMove = 0;
    int legal = 0;
    int origAlpha = alpha;

    for (int i = 0; i < ml.n; i++) {
        int bi = i;
        for (int j = i + 1; j < ml.n; j++)
            if (ml.score[j] > ml.score[bi]) bi = j;
        std::swap(ml.m[i], ml.m[bi]);
        std::swap(ml.score[i], ml.score[bi]);

        Move m = ml.m[i];
        Undo u;
        makeMove(P, m, u);
        if (isAttacked(P, kingSq(P, P.side ^ 1), P.side)) { unmakeMove(P, m, u); continue; }
        legal++;

        bool isCap   = IS_CAP(m) || IS_EP(m);
        bool isPromo = PROMO(m) != 0;

        int score;
        if (legal == 1) {
            score = -alphaBeta(depth - 1, -beta, -alpha, ply + 1, true);
        } else {
            int r = 0;
            if (depth >= 3 && !isCap && !isPromo && !inCheck && legal > 2)
                r = 1 + (legal > 6) + (depth >= 6);

            score = -alphaBeta(depth - 1 - r, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && r > 0)
                score = -alphaBeta(depth - 1, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && score < beta)
                score = -alphaBeta(depth - 1, -beta, -alpha, ply + 1, true);
        }
        unmakeMove(P, m, u);
        if (stopped) return 0;

        if (score > bestScore) {
            bestScore = score;
            bestMove  = m;
            if (score > alpha) {
                alpha = score;
                pvTable[ply][ply] = m;
                for (int k = ply + 1; k < pvLen[ply + 1]; k++)
                    pvTable[ply][k] = pvTable[ply + 1][k];
                pvLen[ply] = pvLen[ply + 1];

                if (score >= beta) {
                    if (!isCap) {
                        killers[ply][1] = killers[ply][0];
                        killers[ply][0] = m;
                        historyTab[FROM(m)][TO(m)] += depth * depth;
                    }
                    break;
                }
            }
        }
    }

    if (legal == 0) return inCheck ? -MATE + ply : 0;

    int flag = (bestScore >= beta) ? TT_BETA
             : (bestScore > origAlpha) ? TT_EXACT : TT_ALPHA;
    ttStore(P.hash, depth, bestScore, bestMove, flag, ply);
    return bestScore;
}

// --------------------------- helpers for UCI --------------------------------
std::string moveToUci(Move m) {
    if (m == 0) return "0000";
    std::string s;
    s += char('a' + (FROM(m) & 7));
    s += char('1' + (FROM(m) >> 3));
    s += char('a' + (TO(m) & 7));
    s += char('1' + (TO(m) >> 3));
    if (PROMO(m)) s += " nbrq"[PROMO(m)];
    return s;
}

Move uciToMove(Position& p, const std::string& str) {
    MoveList ml;
    genMoves(p, ml, false);
    for (int i = 0; i < ml.n; i++) {
        Move m = ml.m[i];
        if (moveToUci(m) != str) continue;
        Undo u;
        makeMove(p, m, u);
        bool ok = !isAttacked(p, kingSq(p, p.side ^ 1), p.side);
        unmakeMove(p, m, u);
        if (ok) return m;
    }
    return 0;
}

U64 perft(int depth) {
    if (depth == 0) return 1;
    MoveList ml;
    genMoves(P, ml, false);
    U64 n = 0;
    for (int i = 0; i < ml.n; i++) {
        Undo u;
        makeMove(P, ml.m[i], u);
        if (isAttacked(P, kingSq(P, P.side ^ 1), P.side)) { unmakeMove(P, ml.m[i], u); continue; }
        n += (depth == 1) ? 1 : perft(depth - 1);
        unmakeMove(P, ml.m[i], u);
    }
    return n;
}

// --------------------------------- search -----------------------------------
void search(int maxTimeMs, int maxDepth) {
    stopped = false;
    nodes = 0;
    startTime = std::chrono::steady_clock::now();
    timeLimitMs = maxTimeMs;

    memset(killers, 0, sizeof(killers));
    memset(historyTab, 0, sizeof(historyTab));
    memset(pvLen, 0, sizeof(pvLen));

    Move bestMove = 0;
    int  bestScore = 0;

    for (int d = 1; d <= maxDepth; d++) {
        int score = alphaBeta(d, -INF, INF, 0, true);
        if (stopped) break;

        if (pvLen[0] > 0) bestMove = pvTable[0][0];
        bestScore = score;

        std::cout << "info depth " << d << " score ";
        if (std::abs(score) > MATE_BOUND) {
            int mateIn = (score > 0) ? (MATE - score + 1) / 2
                                     : -(MATE + score + 1) / 2;
            std::cout << "mate " << mateIn;
        } else {
            std::cout << "cp " << score;
        }
        std::cout << " nodes " << nodes << " time " << elapsedMs() << " pv";
        for (int i = 0; i < pvLen[0]; i++)
            std::cout << " " << moveToUci(pvTable[0][i]);
        std::cout << std::endl;

        if (timeLimitMs > 0 && elapsedMs() > timeLimitMs / 2) break;
    }

    bestRootMove = bestMove;
}

// ---------------------------------- main ------------------------------------
int main() {
    init();
    ttInit(64);
    setStartPos(P);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string token;
        if (!(is >> token)) continue;

        if (token == "uci") {
            std::cout << "id name SimpleEngine 1.0" << std::endl;
            std::cout << "id author You" << std::endl;
            std::cout << "option name Hash type spin default 64 min 1 max 2048" << std::endl;
            std::cout << "uciok" << std::endl;
        }
        else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        }
        else if (token == "ucinewgame") {
            ttClear();
            setStartPos(P);
        }
        else if (token == "setoption") {
            std::string name, value, tok;
            is >> tok;
            while (is >> tok && tok != "value") name += tok + " ";
            is >> value;
            if (name.find("Hash") != std::string::npos && !value.empty())
                ttInit(std::atoi(value.c_str()));
        }
        else if (token == "position") {
            std::string t;
            is >> t;
            if (t == "startpos") {
                setStartPos(P);
                if (is >> t) { /* may be "moves" */ }
            } else if (t == "fen") {
                std::string fenStr;
                while (is >> t && t != "moves") fenStr += t + " ";
                parseFen(P, fenStr);
            }
            if (t == "moves") {
                std::string mv;
                while (is >> mv) {
                    Move m = uciToMove(P, mv);
                    if (!m) break;
                    Undo u;
                    makeMove(P, m, u);
                }
            }
        }
        else if (token == "go") {
            int wtime = 0, btime = 0, winc = 0, binc = 0;
            int movestogo = 0, depth = 64, movetime = 0;
            std::string t;
            while (is >> t) {
                if      (t == "wtime")     is >> wtime;
                else if (t == "btime")     is >> btime;
                else if (t == "winc")      is >> winc;
                else if (t == "binc")      is >> binc;
                else if (t == "movestogo") is >> movestogo;
                else if (t == "depth")     is >> depth;
                else if (t == "movetime")  is >> movetime;
                else if (t == "infinite")  depth = MAX_PLY - 2;
            }
            int limit = 0;
            if (movetime > 0) limit = movetime;
            else if (wtime > 0 || btime > 0) {
                int myTime = (P.side == WHITE) ? wtime : btime;
                int myInc  = (P.side == WHITE) ? winc  : binc;
                if (movestogo > 0) limit = myTime / movestogo + myInc / 2;
                else               limit = myTime / 30 + myInc / 2;
                if (limit < 1) limit = 1;
            }
            search(limit, depth);
            std::cout << "bestmove " << moveToUci(bestRootMove) << std::endl;
        }
        else if (token == "stop") {
            stopped = true;
        }
        else if (token == "perft") {
            int d = 1;
            is >> d;
            auto t0 = std::chrono::steady_clock::now();
            U64 n = perft(d);
            auto t1 = std::chrono::steady_clock::now();
            int ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::cout << "perft " << d << " = " << n << " (" << ms << " ms)" << std::endl;
        }
        else if (token == "d") {
            std::cout << "side: " << (P.side == WHITE ? "white" : "black")
                      << " castling: " << P.castling
                      << " ep: " << P.ep
                      << " hash: " << std::hex << P.hash << std::dec << std::endl;
        }
        else if (token == "quit") {
            break;
        }
    }
    return 0;
}