// Dump the residual (all hyper-connection streams) after every layer at the LAST prompt token.
// usage: PROMPTS=file OUT=file lasttok -m model.gguf [common llama.cpp args, e.g. -ngl 99 --n-cpu-moe 48]
// PROMPTS: one prompt per line, "\n" escaped, chat template already applied (special tokens parsed).
// OUT: per prompt, n_layer x n_hc x n_embd float32, prompts in file order.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct cb_state {
    int n_layer = 0, n_hc = 0, n_embd = 0;
    std::vector<float> buf;          // n_layer * n_hc * n_embd
    std::vector<char> seen;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * s = (cb_state *) ud;
    // exactly "l_last-<il>" (views like "l_last-46 (reshaped)" also match the prefix)
    bool hit = strncmp(t->name, "l_last-", 7) == 0;
    for (const char * c = t->name + 7; hit && *c; c++) hit = *c >= '0' && *c <= '9';
    if (ask) return hit;
    if (!hit) return true;
    const int il = atoi(t->name + 7);
    if (il < 0 || il >= s->n_layer) return true;
    // [n_embd, n_hc, n_rows]; the last row is the last token
    const int64_t n_embd = t->ne[0], n_hc = t->ne[1], n_rows = t->ne[2];
    if (n_embd != s->n_embd || n_hc != s->n_hc || t->type != GGML_TYPE_F32) {
        fprintf(stderr, "unexpected %s shape %lld %lld %lld type %d\n", t->name, (long long) n_embd, (long long) n_hc,
                (long long) n_rows, t->type);
        return false;
    }
    std::vector<float> tmp(n_embd);
    for (int h = 0; h < n_hc; h++) {
        const size_t off = (size_t) (n_rows - 1) * t->nb[2] + (size_t) h * t->nb[1];
        ggml_backend_tensor_get(t, tmp.data(), off, n_embd * sizeof(float));
        memcpy(&s->buf[((size_t) il * n_hc + h) * n_embd], tmp.data(), n_embd * sizeof(float));
    }
    s->seen[il] = 1;
    return true;
}

static std::string unescape(const std::string & in) {
    std::string out;
    for (size_t i = 0; i < in.size(); i++) {
        if (in[i] == '\\' && i + 1 < in.size() && in[i + 1] == 'n') { out += '\n'; i++; }
        else out += in[i];
    }
    return out;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    const char * pf = getenv("PROMPTS"), * of = getenv("OUT");
    const int n_hc = getenv("N_HC") ? atoi(getenv("N_HC")) : 4;
    if (!pf || !of) { fprintf(stderr, "set PROMPTS and OUT\n"); return 1; }

    cb_state st;
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    common_init();
    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { fprintf(stderr, "load failed\n"); return 1; }
    st.n_layer = llama_model_n_layer(model);
    st.n_embd = llama_model_n_embd(model);
    st.n_hc = n_hc;
    st.buf.assign((size_t) st.n_layer * n_hc * st.n_embd, 0.f);
    st.seen.assign(st.n_layer, 0);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::ifstream in(pf);
    FILE * out = fopen(of, "wb");
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<llama_token> toks = common_tokenize(vocab, unescape(line), false, true);
        llama_memory_clear(llama_get_memory(ctx), true);
        std::fill(st.seen.begin(), st.seen.end(), 0);
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), toks.size()))) { fprintf(stderr, "decode failed\n"); return 1; }
        llama_synchronize(ctx);
        for (int il = 0; il < st.n_layer; il++)
            if (!st.seen[il]) { fprintf(stderr, "layer %d not captured\n", il); return 1; }
        fwrite(st.buf.data(), sizeof(float), st.buf.size(), out);
        fprintf(stderr, "prompt %d: %zu tokens\n", ++n, toks.size());
    }
    fclose(out);
    fprintf(stderr, "done: %d prompts, layout %d x %d x %d f32\n", n, st.n_layer, st.n_hc, st.n_embd);
    return 0;
}
