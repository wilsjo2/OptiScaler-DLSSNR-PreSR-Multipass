#pragma once
// Slider calibrator, shared by the Display Filter and the in-game menu.
//
// It makes one model pass look like the output of a saved preset (the goal) on one held frame by
// trying values of five sliders: the pass's Intensity, Local structure and Local tone, and the Detail strength and
// Colour strength of the composition. On request it also tries the four skin / environment sliders of the composition
// and the Highlight guard. The host renders and reads pictures back; this file only compares two pictures (Reader,
// Difference) and chooses the values to try next (Solver).
#include <dxgiformat.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

namespace NrCalibrate
{
enum Value : unsigned
{
    Intensity,
    Structure,
    Tone,
    Detail,
    Colour,
    // The optional ones ("skin settings"): the composition's separate skin / environment sliders and Highlight guard.
    SkinDetail,
    SkinColour,
    EnvironmentDetail,
    EnvironmentColour,
    Guard,
    Count
};
inline constexpr unsigned BasicCount = SkinDetail; // the five that are always searched
using Point = std::array<float, Count>;
inline constexpr std::array<const char*, Count> Names { "Intensity",         "Local structure",   "Local tone",
                                                        "Detail strength",   "Colour strength",   "Skin detail",
                                                        "Skin colour",       "Environment detail", "Environment colour",
                                                        "Highlight guard" };
// The sliders' own ranges.
inline constexpr Point Lowest { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, Highest { 2, 2, 2, 2, 4, 1, 1, 1, 1, 8 };

// A picture in an encoding where a step is about equally visible in dark and bright areas, three values a pixel.
struct Picture
{
    unsigned width = 0, height = 0;
    std::vector<float> values;
    bool Empty() const { return values.empty(); }
    // Adds another picture of the same size (to average several frames); false when the sizes differ.
    bool Add(const Picture& other)
    {
        if (values.empty())
        {
            *this = other;
            return true;
        }
        if (other.width != width || other.height != height)
            return false;
        for (size_t i = 0; i < values.size(); ++i)
            values[i] += other.values[i];
        return true;
    }
    void Scale(float factor)
    {
        for (auto& value : values)
            value *= factor;
    }
};

// Mean squared difference of two pictures; negative when they cannot be compared.
inline double Difference(const Picture& a, const Picture& b)
{
    if (a.Empty() || a.width != b.width || a.height != b.height || a.values.size() != b.values.size())
        return -1.0;
    double sum = 0;
    for (size_t i = 0; i < a.values.size(); ++i)
    {
        const double d = double(a.values[i]) - double(b.values[i]);
        sum += d * d;
    }
    return sum / double(a.values.size());
}

// The same difference as a likeness in decibels (higher = more alike; +6 dB = half the error).
inline double Decibels(double difference) { return -10.0 * std::log10(std::max(difference, 1e-12)); }

// How much each pixel of a Picture counts as skin (0 = none, 1 = skin), read from the app's own skin mask picture
// (Reader::ReadMask). With it the skin part and the rest (the environment) are compared separately: skin is a small
// part of most pictures, so in a difference over the whole picture the skin sliders hardly count.
struct SkinMask
{
    unsigned width = 0, height = 0;
    std::vector<float> skin; // one value a pixel
    double share = 0;        // the part of the picture that is skin (the mean of `skin`)
    bool Fits(const Picture& picture) const
    {
        return !skin.empty() && width == picture.width && height == picture.height &&
               picture.values.size() == skin.size() * 3;
    }
    // Enough of both parts to judge each on its own.
    bool Usable() const { return !skin.empty() && share >= 0.005 && share <= 0.995; }
};

// Mean squared difference of two pictures over the skin part and over the environment, each weighted by the mask.
struct Regions
{
    double skin = -1, environment = -1;
    bool Valid() const { return skin >= 0 && environment >= 0; }
    // What the search lowers: both parts count the same, whatever their size.
    double Balanced() const { return 0.5 * (skin + environment); }
};

inline Regions Difference(const Picture& a, const Picture& b, const SkinMask& mask)
{
    if (a.Empty() || a.width != b.width || a.height != b.height || a.values.size() != b.values.size() || !mask.Fits(a))
        return {};
    double skin = 0, environment = 0, skinWeight = 0, environmentWeight = 0;
    for (size_t pixel = 0; pixel < mask.skin.size(); ++pixel)
    {
        double sum = 0;
        for (size_t c = 0; c < 3; ++c)
        {
            const double d = double(a.values[pixel * 3 + c]) - double(b.values[pixel * 3 + c]);
            sum += d * d;
        }
        const double w = mask.skin[pixel];
        skin += w * sum;
        environment += (1.0 - w) * sum;
        skinWeight += w;
        environmentWeight += 1.0 - w;
    }
    if (skinWeight <= 0 || environmentWeight <= 0)
        return {};
    return { skin / (3.0 * skinWeight), environment / (3.0 * environmentWeight) };
}

// A change of a model slider restarts the model's history, and on a held frame the picture then keeps changing for
// some frames. Measured too early, such a picture is judged against one in another state; the error is small, but so
// is what the skin sliders change. Give every picture read since the change: Add() says when one can be measured.
class Settling
{
  public:
    // `most`: pictures at most; the last one is then taken as it is.
    void Begin(unsigned most = 8)
    {
        previous = {};
        last = -1;
        count = 0;
        limit = std::max(most, 1u);
    }
    // `scale`: the difference this picture is about to be judged by (its difference from the goal). True when
    //   the picture differs from the one before by less than 1/2000 of that (what is left of the change then moves the
    //   result by about a percent at most),
    //   the change between pictures has stopped shrinking (what is left is noise, not settling), or
    //   `most` pictures were read.
    bool Add(const Picture& picture, double scale)
    {
        ++count;
        const double change = previous.Empty() ? -1.0 : Difference(previous, picture);
        const double before = last;
        previous = picture;
        last = change;
        if (count >= limit)
            return true;
        if (change < 0)
            return false; // the first picture, or one of another size
        if (change <= scale / 2000.0)
            return true;
        return before >= 0 && change > before * 0.5;
    }
    // The difference of the last two pictures; negative when fewer than two were read.
    double Change() const { return last; }
    unsigned Pictures() const { return count; }

  private:
    Picture previous;
    double last = -1;
    unsigned count = 0, limit = 8;
};

// Turns the rows of a texture read back from the GPU into a Picture. Integer formats are display-encoded already and
// are taken as they are. Float formats hold linear light, of one of two kinds (`sceneLight`, set before the first
// picture is read):
//   false: a finished picture in scRGB (1.0 = 80 nits), as the Display Filter outputs. It is encoded with the PQ
//          curve, which is made for equal visibility of steps at every brightness, SDR and HDR alike.
//   true:  a game's picture before its tone mapping, of unknown scale. The first picture read (the goal's) sets
//          an exposure so that its mean becomes mid grey, every later picture is encoded with that same exposure,
//          and a curve that flattens towards the highlights follows.
class Reader
{
  public:
    bool sceneLight = false;

    static bool Readable(DXGI_FORMAT format)
    {
        switch (format)
        {
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
        case DXGI_FORMAT_R11G11B10_FLOAT:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return true;
        default:
            return false;
        }
    }

    void Reset() { scale = 0; }

    // About 600,000 pixels are kept at most (plenty to compare whole-picture sliders): every step-th pixel in both
    // directions.
    bool Read(Picture& out, DXGI_FORMAT format, const void* data, unsigned width, unsigned height, unsigned rowPitch)
    {
        if (!Readable(format) || !data || !width || !height)
            return false;
        const unsigned step =
            std::max(1u, unsigned(std::ceil(std::sqrt(double(width) * double(height) / 600000.0))));
        out.width = (width + step - 1) / step;
        out.height = (height + step - 1) / step;
        out.values.resize(size_t(out.width) * out.height * 3);
        const bool linear = format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R32G32B32A32_FLOAT ||
                            format == DXGI_FORMAT_R11G11B10_FLOAT;
        if (linear && scale <= 0 && !sceneLight)
        {
            scale = 1.0f;
            BuildTables();
        }
        if (linear && scale <= 0)
        {
            // Exposure from this first picture: its mean (negative and non-finite values count as black).
            double sum = 0;
            Visit(format, data, width, height, rowPitch, step, [&](size_t, float r, float g, float b)
                  { sum += double(Clean(r)) + double(Clean(g)) + double(Clean(b)); });
            const double mean = sum / double(out.values.size());
            scale = mean > 1e-6 ? float(0.18 / mean) : 1.0f;
            BuildTables();
        }
        float* values = out.values.data();
        if (format == DXGI_FORMAT_R16G16B16A16_FLOAT)
        {
            const auto* bytes = static_cast<const uint8_t*>(data);
            size_t i = 0;
            for (unsigned y = 0; y < height; y += step)
            {
                const auto* row = reinterpret_cast<const uint16_t*>(bytes + size_t(y) * rowPitch);
                for (unsigned x = 0; x < width; x += step, i += 3)
                {
                    values[i] = half[row[x * 4]];
                    values[i + 1] = half[row[x * 4 + 1]];
                    values[i + 2] = half[row[x * 4 + 2]];
                }
            }
        }
        else if (linear)
            Visit(format, data, width, height, rowPitch, step, [&](size_t i, float r, float g, float b)
                  {
                      values[i] = Encode(r);
                      values[i + 1] = Encode(g);
                      values[i + 2] = Encode(b);
                  });
        else
            Visit(format, data, width, height, rowPitch, step, [&](size_t i, float r, float g, float b)
                  {
                      values[i] = r;
                      values[i + 1] = g;
                      values[i + 2] = b;
                  });
        return true;
    }

    // The app's skin mask picture (white where the composition treats a pixel as skin), thinned exactly as Read()
    // thins a picture. The mask is drawn at the picture's white level, which is not known here: the brightest pixel
    // is taken as 1. A picture without any skin gives a mask of zeros.
    static bool ReadMask(SkinMask& out, DXGI_FORMAT format, const void* data, unsigned width, unsigned height,
                         unsigned rowPitch)
    {
        if (!Readable(format) || !data || !width || !height)
            return false;
        const unsigned step =
            std::max(1u, unsigned(std::ceil(std::sqrt(double(width) * double(height) / 600000.0))));
        out.width = (width + step - 1) / step;
        out.height = (height + step - 1) / step;
        out.skin.assign(size_t(out.width) * out.height, 0.0f);
        float peak = 0;
        Visit(format, data, width, height, rowPitch, step, [&](size_t i, float r, float, float)
              {
                  const float value = std::isfinite(r) && r > 0 ? r : 0.0f;
                  out.skin[i / 3] = value;
                  peak = std::max(peak, value);
              });
        double sum = 0;
        for (auto& value : out.skin)
        {
            value = peak > 1e-6f ? std::min(value / peak, 1.0f) : 0.0f;
            sum += value;
        }
        out.share = sum / double(out.skin.size());
        return true;
    }

  private:
    float scale = 0;
    std::vector<float> half; // encoded value of every 16-bit float

    static float Clean(float value) { return std::isfinite(value) && value > 0 ? value : 0.0f; }

    float Encode(float value) const
    {
        if (std::isnan(value) || value <= 0)
            return 0.0f;
        if (std::isinf(value))
            return 1.0f;
        if (sceneLight)
        {
            const float x = value * scale;
            return std::pow(x / (1.0f + x), 1.0f / 2.2f);
        }
        // PQ (SMPTE ST 2084) of the luminance in nits; 10,000 nits and above encode as 1.
        const float y = std::min(value * 80.0f / 10000.0f, 1.0f);
        const float p = std::pow(y, 0.1593017578125f);
        return std::pow((0.8359375f + 18.8515625f * p) / (1.0f + 18.6875f * p), 78.84375f);
    }

    static float HalfToFloat(uint16_t h)
    {
        const uint32_t sign = uint32_t(h & 0x8000u) << 16;
        uint32_t exponent = (h >> 10) & 0x1Fu, mantissa = h & 0x3FFu, bits;
        if (exponent == 0)
        {
            if (mantissa == 0)
                bits = sign;
            else
            {
                exponent = 1;
                while (!(mantissa & 0x400u))
                {
                    mantissa <<= 1;
                    --exponent;
                }
                mantissa &= 0x3FFu;
                bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
            }
        }
        else if (exponent == 31)
            bits = sign | 0x7F800000u | (mantissa << 13);
        else
            bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    // An unsigned small float of R11G11B10_FLOAT: 5 exponent bits and `mantissaBits` (6 or 5) mantissa bits.
    static float SmallFloat(uint32_t value, unsigned mantissaBits)
    {
        const uint32_t exponent = value >> mantissaBits, mantissa = value & ((1u << mantissaBits) - 1u);
        const float fraction = float(mantissa) / float(1u << mantissaBits);
        if (exponent == 0)
            return std::ldexp(fraction, -14);
        if (exponent == 31)
            return mantissa ? 0.0f : std::numeric_limits<float>::infinity();
        return std::ldexp(1.0f + fraction, int(exponent) - 15);
    }

    void BuildTables()
    {
        half.resize(65536);
        for (uint32_t h = 0; h < 65536; ++h)
            half[h] = Encode(HalfToFloat(uint16_t(h)));
    }

    template <class F>
    static void Visit(DXGI_FORMAT format, const void* data, unsigned width, unsigned height, unsigned rowPitch,
                      unsigned step, F&& pixel)
    {
        const auto* bytes = static_cast<const uint8_t*>(data);
        size_t i = 0;
        for (unsigned y = 0; y < height; y += step)
        {
            const uint8_t* row = bytes + size_t(y) * rowPitch;
            for (unsigned x = 0; x < width; x += step, i += 3)
            {
                switch (format)
                {
                case DXGI_FORMAT_R16G16B16A16_FLOAT:
                {
                    const auto* p = reinterpret_cast<const uint16_t*>(row) + size_t(x) * 4;
                    pixel(i, HalfToFloat(p[0]), HalfToFloat(p[1]), HalfToFloat(p[2]));
                    break;
                }
                case DXGI_FORMAT_R32G32B32A32_FLOAT:
                {
                    const auto* p = reinterpret_cast<const float*>(row) + size_t(x) * 4;
                    pixel(i, p[0], p[1], p[2]);
                    break;
                }
                case DXGI_FORMAT_R11G11B10_FLOAT:
                {
                    uint32_t v;
                    std::memcpy(&v, row + size_t(x) * 4, 4);
                    pixel(i, SmallFloat(v & 0x7FFu, 6), SmallFloat((v >> 11) & 0x7FFu, 6), SmallFloat(v >> 22, 5));
                    break;
                }
                case DXGI_FORMAT_R10G10B10A2_UNORM:
                {
                    uint32_t v;
                    std::memcpy(&v, row + size_t(x) * 4, 4);
                    pixel(i, float(v & 0x3FFu) / 1023.0f, float((v >> 10) & 0x3FFu) / 1023.0f,
                          float((v >> 20) & 0x3FFu) / 1023.0f);
                    break;
                }
                case DXGI_FORMAT_B8G8R8A8_UNORM:
                case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
                {
                    const uint8_t* p = row + size_t(x) * 4;
                    pixel(i, p[2] / 255.0f, p[1] / 255.0f, p[0] / 255.0f);
                    break;
                }
                default: // R8G8B8A8
                {
                    const uint8_t* p = row + size_t(x) * 4;
                    pixel(i, p[0] / 255.0f, p[1] / 255.0f, p[2] / 255.0f);
                    break;
                }
                }
            }
        }
    }
};

// A search over some of the sliders (Options::mask), one value at a time; the caller measures each and reports its
// loss (lower = more alike). One slider is varied while the others stay.
//   Scan rounds (`scans` of them): every slider in turn is tried at each point of a coarse grid over its whole range,
//   and its best value is kept. The picture's response to a slider has several dips, so a search that only looks
//   near the starting value ends in whichever dip is nearest.
//   Fine rounds: a step up and a step down are measured, a parabola through the three results suggests where the
//   lowest point lies, and the best value measured is kept. A slider whose best value stayed inside its step gets
//   half the step next round; the search ends when every step is below `finest`, or after `budget` measurements.
//   After each fine round the whole change of that round is tried once more, all sliders together (`pattern`), and
//   again while it keeps helping: sliders that must move together to improve the picture are followed that way,
//   where one slider at a time would stop.
class Search
{
  public:
    struct Options
    {
        Point low = Lowest, high = Highest;
        // Spacing of the scan; the fine rounds start at half of it.
        Point grid { 0.25f, 0.25f, 0.25f, 0.25f, 0.5f, 0.25f, 0.25f, 0.25f, 0.25f, 1.0f };
        unsigned scans = 2;
        Point finest { 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.1f };
        unsigned budget = 180;
        unsigned mask = (1u << Count) - 1u; // bit n clear: slider n is left alone
        bool pattern = true;
    };

    void Start(const Point& start, const Options& choices)
    {
        options = choices;
        for (unsigned i = 0; i < Count; ++i)
        {
            best[i] = Round(std::clamp(start[i], options.low[i], options.high[i]));
            step[i] = options.grid[i] * 0.5f;
        }
        trial = roundStart = best;
        stage = Stage::First;
        trials = round = patternMoves = 0;
        bestLoss = firstLoss = 0;
        done = false;
    }

    bool Done() const { return done; }
    const Point& Trial() const { return trial; } // the values to measure next
    const Point& Best() const { return best; }
    double BestLoss() const { return bestLoss; }
    double FirstLoss() const { return firstLoss; } // the loss of the starting values
    unsigned Trials() const { return trials; }
    unsigned Budget() const { return options.budget; }

    // The loss measured with Trial().
    void Report(double loss)
    {
        if (done)
            return;
        ++trials;
        switch (stage)
        {
        case Stage::First:
            bestLoss = firstLoss = loss;
            axis = Count; // NextAxis() moves to the first slider in use
            if (!NextAxis())
            {
                done = true;
                return;
            }
            round = 0;
            Begin();
            return;
        case Stage::Scan:
            if (loss < scanLoss)
            {
                scanLoss = loss;
                scanValue = trial[axis];
            }
            if (++scanIndex < scanPoints.size() && trials < options.budget)
            {
                SetTrial(scanPoints[scanIndex]);
                return;
            }
            best[axis] = scanValue;
            bestLoss = scanLoss;
            Advance();
            return;
        case Stage::Plus:
            plusLoss = loss;
            if (minus)
            {
                stage = Stage::Minus;
                SetTrial(*minus);
                return;
            }
            break;
        case Stage::Minus:
            minusLoss = loss;
            if (plus && Fit())
            {
                stage = Stage::Fit;
                SetTrial(*fit);
                return;
            }
            break;
        case Stage::Fit:
            fitLoss = loss;
            break;
        case Stage::Pattern:
            if (loss < bestLoss)
            {
                roundStart = best;
                best = trial;
                bestLoss = loss;
                if (++patternMoves < 4 && trials < options.budget && StartPattern())
                    return;
            }
            patternMoves = 0;
            roundStart = best;
            Begin();
            return;
        }
        Decide();
    }

  private:
    enum class Stage
    {
        First,
        Scan,
        Plus,
        Minus,
        Fit,
        Pattern
    };
    Options options;
    Point best {}, trial {}, step {}, roundStart {}; // roundStart: the best values when the current round began
    unsigned patternMoves = 0;
    double bestLoss = 0, firstLoss = 0, plusLoss = 0, minusLoss = 0, fitLoss = 0, scanLoss = 0;
    float scanValue = 0;
    std::vector<float> scanPoints;
    size_t scanIndex = 0;
    std::optional<float> plus, minus, fit;
    Stage stage = Stage::First;
    unsigned axis = 0, trials = 0, round = 0; // round: complete passes over the sliders so far
    bool done = false;

    static float Round(float value) { return std::round(value * 100.0f) / 100.0f; }
    void SetTrial(float value)
    {
        trial = best;
        trial[axis] = value;
    }

    // The next slider in use after `axis`; false when none is in use. Coming back to the first ends a round.
    bool NextAxis()
    {
        for (unsigned n = 0; n < Count; ++n)
        {
            axis = axis >= Count - 1 ? 0 : axis + 1;
            if (options.mask >> axis & 1u)
                return true;
        }
        return false;
    }

    unsigned FirstAxis() const
    {
        for (unsigned i = 0; i < Count; ++i)
            if (options.mask >> i & 1u)
                return i;
        return 0;
    }

    void Advance()
    {
        NextAxis();
        if (axis == FirstAxis() && RoundEnded())
            return;
        Begin();
    }

    // A pass over the sliders is complete; true when a pattern trial was started.
    bool RoundEnded()
    {
        ++round;
        if (options.pattern && round > options.scans && trials < options.budget && StartPattern())
            return true;
        roundStart = best;
        return false;
    }

    // The change from `roundStart` to the best values, applied once more; false when that is no new point.
    bool StartPattern()
    {
        Point next = best;
        bool moved = false;
        for (unsigned i = 0; i < Count; ++i)
            if (options.mask >> i & 1u)
            {
                next[i] = Round(std::clamp(best[i] + (best[i] - roundStart[i]), options.low[i], options.high[i]));
                moved = moved || std::fabs(next[i] - best[i]) > 0.004f;
            }
        if (!moved)
            return false;
        trial = next;
        stage = Stage::Pattern;
        return true;
    }

    bool Finished() const
    {
        if (trials >= options.budget)
            return true;
        if (round < options.scans)
            return false;
        for (unsigned i = 0; i < Count; ++i)
            if ((options.mask >> i & 1u) && step[i] >= options.finest[i])
                return false;
        return true;
    }

    // Starts on slider `axis`. A slider with nothing to try is passed over (in a fine round with a smaller step).
    void Begin()
    {
        for (unsigned guard = 0; guard < 8 * Count; ++guard)
        {
            if (Finished())
            {
                done = true;
                trial = best;
                return;
            }
            const float centre = best[axis], h = step[axis];
            if (round < options.scans)
            {
                scanPoints.clear();
                const float spacing = std::max(options.grid[axis], 0.01f);
                for (float x = options.low[axis]; x <= options.high[axis] + 0.004f; x += spacing)
                    if (const float value = Round(std::min(x, options.high[axis])); std::fabs(value - centre) > 0.004f)
                        scanPoints.push_back(value);
                if (!scanPoints.empty())
                {
                    stage = Stage::Scan;
                    scanIndex = 0;
                    scanValue = centre;
                    scanLoss = bestLoss;
                    SetTrial(scanPoints[0]);
                    return;
                }
            }
            else
            {
                plus.reset();
                minus.reset();
                fit.reset();
                const float up = Round(std::min(centre + h, options.high[axis]));
                const float down = Round(std::max(centre - h, options.low[axis]));
                if (up > centre + 0.004f)
                    plus = up;
                if (down < centre - 0.004f)
                    minus = down;
                if (plus)
                {
                    stage = Stage::Plus;
                    SetTrial(*plus);
                    return;
                }
                if (minus)
                {
                    stage = Stage::Minus;
                    SetTrial(*minus);
                    return;
                }
                step[axis] = h * 0.5f;
            }
            NextAxis();
            if (axis == FirstAxis() && RoundEnded())
                return;
        }
        done = true;
        trial = best;
    }

    // The lowest point of the parabola through the three measurements, when it curves upward and is a new value.
    bool Fit()
    {
        const double c = best[axis], xp = *plus, xm = *minus;
        const double slopeUp = (plusLoss - bestLoss) / (xp - c), slopeDown = (bestLoss - minusLoss) / (c - xm);
        const double curve = (slopeUp - slopeDown) / (xp - xm); // half the second derivative
        if (!(curve > 1e-12))
            return false;
        // Slope at the centre: the mean of the two sides, weighted so that it is exact for a parabola.
        const double slope = (slopeUp * (c - xm) + slopeDown * (xp - c)) / (xp - xm);
        const double h = step[axis];
        double x = c - slope / (2.0 * curve);
        x = std::clamp(x, c - 2.0 * h, c + 2.0 * h);
        x = std::clamp(x, double(options.low[axis]), double(options.high[axis]));
        const float value = Round(float(x));
        if (std::fabs(value - c) < 0.004 || std::fabs(value - xp) < 0.004 || std::fabs(value - xm) < 0.004)
            return false;
        fit = value;
        return true;
    }

    void Decide()
    {
        const float centre = best[axis], h = step[axis];
        float value = centre;
        double loss = bestLoss;
        const auto consider = [&](const std::optional<float>& x, double measured)
        {
            if (x && measured < loss)
            {
                value = *x;
                loss = measured;
            }
        };
        consider(plus, plusLoss);
        consider(minus, minusLoss);
        consider(fit, fitLoss);
        best[axis] = value;
        bestLoss = loss;
        // A best value a whole step away (or further) may lie further still: keep the step. Otherwise halve it.
        if (std::fabs(value - centre) < h * 0.999f)
            step[axis] = h * 0.5f;
        Advance();
    }
};

// The calibrator's search. The sliders are of two kinds. Intensity, Local structure and Local tone go into the
// model: the picture answers them unevenly, and with Detail strength at 0 they change nothing at all, so judged on
// their own a search can end somewhere poor. The others only blend the model's edit into the picture: the picture
// answers them smoothly and a change needs no new model set-up. So:
//   1. The model sliders are searched (two scan rounds, then fine rounds), and every value tried for them is judged at
//      the best Detail and Colour strength found for it by a short search of those two.
//   2. With `skin` set: the skin / environment sliders and Highlight guard are searched over their whole ranges,
//      together with Detail and Colour strength, at the model values found.
//   3. All sliders in use are searched together, finely, from the best values found (the short search of step 1 is
//      too coarse to show the last small gains).
class Solver
{
  public:
    struct Options
    {
        Point low = Lowest, high = Highest;
        bool skin = false;      // also search SkinDetail .. Guard; otherwise they stay at their starting values
        unsigned budget = 2500; // measurements at most
    };

    void Start(const Point& start, const Options& choices)
    {
        options = choices;
        Search::Options model;
        model.low = options.low;
        model.high = options.high;
        model.mask = 1u << Intensity | 1u << Structure | 1u << Tone;
        model.budget = 120;
        outer.Start(start, model);
        best = outer.Best();
        bestLoss = firstLoss = 0;
        trials = modelTrials = 0;
        first = true;
        done = false;
        stage = Stage::Model;
        BeginInner();
    }

    bool Done() const { return done; }
    const Point& Trial() const { return inner.Trial(); } // the values to measure next
    const Point& Best() const { return best; }
    double BestLoss() const { return bestLoss; }
    double FirstLoss() const { return firstLoss; } // the loss of the starting values
    unsigned Trials() const { return trials; }
    unsigned ModelTrials() const { return modelTrials; } // sets of model slider values judged so far
    bool Skin() const { return options.skin; }
    // Sliders in use: the first five, or all.
    unsigned Sliders() const { return options.skin ? unsigned(Count) : BasicCount; }

    // The loss measured with Trial().
    void Report(double loss)
    {
        if (done)
            return;
        ++trials;
        const Point measured = inner.Trial();
        if (first)
            firstLoss = loss;
        inner.Report(loss);
        // The model stage keeps the best of everything it measured. The stages after it start by measuring those
        // values again and move only to what measures better than that: their own search holds the answer. A loss of
        // an earlier stage is not compared with theirs, because the pictures of the model stage are read soon after
        // a restart of the model's history and those of the later stages long after one.
        if (stage == Stage::Model)
        {
            if (first || loss < bestLoss)
            {
                bestLoss = loss;
                best = measured;
            }
        }
        else
        {
            best = inner.Best();
            bestLoss = inner.BestLoss();
        }
        first = false;
        if (!inner.Done() && trials < options.budget)
            return;
        if (stage == Stage::Polish || trials >= options.budget)
        {
            done = true;
            return;
        }
        if (stage == Stage::Skin)
            stage = Stage::Polish;
        else
        {
            ++modelTrials;
            outer.Report(inner.BestLoss());
            if (outer.Done())
                stage = options.skin ? Stage::Skin : Stage::Polish;
        }
        BeginInner();
    }

  private:
    enum class Stage
    {
        Model,
        Skin,
        Polish
    };
    Options options;
    Search outer, inner;
    Point best {};
    double bestLoss = 0, firstLoss = 0;
    unsigned trials = 0, modelTrials = 0;
    bool first = true, done = false;
    Stage stage = Stage::Model;

    void BeginInner()
    {
        Search::Options blend;
        blend.low = options.low;
        blend.high = options.high;
        const unsigned strengths = 1u << Detail | 1u << Colour;
        const unsigned skin = (1u << Count) - (1u << BasicCount);
        if (stage == Stage::Model)
        {
            // The two strengths, from the best pair so far, for the model values now being tried. One round: a
            // parabola fits these two well.
            blend.mask = strengths;
            blend.scans = 0;
            blend.grid.fill(0.4f); // (the first step is half of it)
            blend.finest.fill(0.15f);
            blend.budget = 24;
            blend.pattern = false;
            Point from = outer.Trial();
            from[Detail] = best[Detail];
            from[Colour] = best[Colour];
            inner.Start(from, blend);
            return;
        }
        if (stage == Stage::Skin)
        {
            blend.mask = strengths | skin;
            blend.scans = 2;
            blend.grid[Guard] = 0.5f;
            blend.budget = 450;
        }
        else
        {
            blend.mask = options.skin ? (1u << Count) - 1u : (1u << BasicCount) - 1u;
            blend.scans = 0;
            for (unsigned i = 0; i < Count; ++i)
                blend.grid[i] = i == Guard ? 0.8f : 0.2f;
            blend.budget = options.skin ? 320 : 200;
        }
        inner.Start(best, blend);
    }
};
} // namespace NrCalibrate
