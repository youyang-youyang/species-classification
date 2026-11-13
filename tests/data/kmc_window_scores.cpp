// Build against refresh-bio/KMC v3.2.1 kmc_api (upstream GNU GPL v3).
#include "kmc_file.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
 try {
    if (argc != 4) throw std::runtime_error("usage: query DATABASE SCORES.bin STATS.json; sequences on stdin");
    static_assert(sizeof(float) == 4, "float32 required");
    CKMCFile db;
    if (!db.OpenForRA(argv[1])) throw std::runtime_error("Cannot open database");
    CKMCFileInfo info;
    if (!db.Info(info) || info.kmer_length != 21 || info.both_strands)
        throw std::runtime_error("Expected strand-specific 21-mer database");
    float lookup[256]; lookup[0] = 0;
    for (int c=1; c<256; ++c) lookup[c] = float(std::round(std::exp(-5.0*(c-1)/254)*10000)/10000);
    std::ofstream out(argv[2], std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output");
    std::string seq;
    std::vector<uint32> counts;
    uint64_t windows=0, values=0, invalid=0, absent=0, checked=0;
    auto valid = [](char c){ return c=='A'||c=='C'||c=='G'||c=='T'; };
    while (std::getline(std::cin, seq)) {
        if (!db.GetCountersForRead(seq, counts) || counts.size()!=seq.size()-20)
            throw std::runtime_error("Unexpected read/count length");
        std::vector<float> scores(counts.size());
        int bad=0;
        for (size_t j=0;j<20;++j) bad += !valid(seq[j]);
        for (size_t i=0;i<counts.size();++i) {
            bad += !valid(seq[i+20]);
            if (counts[i]>255) throw std::runtime_error("Counter exceeds 255");
            if (bad) { scores[i]=-1; ++invalid; }
            else {
                scores[i]=lookup[counts[i]];
                absent += counts[i]==0;
                if (windows % 1000 == 0) {
                    CKmerAPI kmer(21); uint32 direct=0;
                    kmer.from_string(seq.substr(i,21));
                    db.CheckKmer(kmer,direct);
                    if(direct!=counts[i]) throw std::runtime_error("Positional query mismatch");
                    ++checked;
                }
            }
            bad -= !valid(seq[i]);
        }
        out.write(reinterpret_cast<const char*>(scores.data()), scores.size()*sizeof(float));
        if(!out) throw std::runtime_error("Output write failed");
        ++windows; values+=scores.size();
        if(windows%10000==0) std::cerr<<"SCORED "<<windows<<" unique windows\n";
    }
    if(std::cin.bad()) throw std::runtime_error("Input read failed");
    out.close();
    if(!out) throw std::runtime_error("Output close failed");
    std::ofstream report(argv[3]);
    report<<"{\"windows\":"<<windows<<",\"values\":"<<values<<",\"invalid\":"<<invalid
          <<",\"absent\":"<<absent<<",\"direct_checks\":"<<checked<<"}\n";
    if(!report) throw std::runtime_error("Report write failed");
    return 0;
 } catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<"\n"; return 1; }
}
