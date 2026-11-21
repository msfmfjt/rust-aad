// Public XAD tape replay for the registered 10,000-path affine32-v1 fixture.
#include <XAD/XAD.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Mode = xad::adj<double>;
using Input = std::array<double, 2>;
using Result = std::array<double, 4>;
constexpr std::size_t paths = 10000;

template<class Row>
std::vector<Row> read_rows(const std::string& file) {
    static_assert(std::endian::native == std::endian::little && sizeof(double) == 8);
    std::vector<Row> rows(paths);
    std::ifstream input(file, std::ios::binary);
    const auto bytes = static_cast<std::streamsize>(rows.size() * sizeof(Row));
    if (!input.read(reinterpret_cast<char*>(rows.data()), bytes) || input.peek() != EOF)
        throw std::runtime_error("incorrect size: " + file);
    return rows;
}

void sweep(Mode::tape_type& tape, const std::vector<Input>& input, std::vector<Result>& out) {
    for (std::size_t i = 0; i < paths; ++i) {
        Mode::active_type x = input[i][0], q = input[i][1];
        tape.registerInput(x);
        tape.registerInput(q);
        tape.newRecording();
        Mode::active_type s = x;
        for (int k = 0; k < 32; ++k) {
            Mode::active_type product = s * q;
            s = product + 1.0 / 64.0;
        }
        Mode::active_type y = s * x;
        tape.registerOutput(s);
        tape.registerOutput(y);
        xad::derivative(s) = 1.0;
        xad::derivative(y) = -0.5;
        tape.computeAdjoints();
        out[i] = {xad::value(s), xad::value(y), xad::derivative(x), xad::derivative(q)};
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: xad-affine REPLAY_DIRECTORY OUTPUT.f64le\n";
        return 2;
    }
    const std::string dir = argv[1];
    const auto input = read_rows<Input>(dir + "/inputs.f64le");
    std::uint64_t digest = 0xcbf29ce484222325ULL;
    for (const auto& row : input)
        for (auto byte : std::bit_cast<std::array<unsigned char, sizeof(Input)>>(row))
            digest = (digest ^ byte) * 0x100000001b3ULL;
    if (digest != 0xa7bf6211c9ab67d9ULL) throw std::runtime_error("wrong affine32-v1 inputs");
    const auto expected = read_rows<Result>(dir + "/expected.f64le");
    Mode::tape_type tape;
    std::vector<Result> observed(paths);
    sweep(tape, input, observed);
    for (std::size_t i = 0; i < paths; ++i)
        for (std::size_t j = 0; j < observed[i].size(); ++j) {
            const double a = observed[i][j], b = expected[i][j];
            if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b) > 2e-13 * (1.0+std::abs(b)))
                throw std::runtime_error("replay mismatch at row " + std::to_string(i));
        }
    std::array<long long, 11> ns{};
    for (auto& sample : ns) {
        const auto start = std::chrono::steady_clock::now();
        sweep(tape, input, observed);
        sample = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
    }
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char*>(observed.data()), paths * sizeof(Result));
    output.close();
    if (!output) throw std::runtime_error("cannot write replay result");
    std::cout << "{\"workload\":\"affine32-v1\",\"backend\":\"xad-tape\",\"paths\":10000,"
                 "\"verified_paths\":10000,\"input_fnv1a64\":\"a7bf6211c9ab67d9\","
                 "\"phase\":\"record_forward_vjp_store\",\"samples_ns\":[";
    for (std::size_t i = 0; i < ns.size(); ++i) std::cout << (i ? "," : "") << ns[i];
    std::sort(ns.begin(), ns.end());
    std::cout << "],\"median_ns\":" << ns[5] << "}\n";
}
