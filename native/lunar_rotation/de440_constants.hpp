#pragma once
// Frozen DE440 constants inherited unchanged from JX-PUBLIC-LUNAR-PROPAGATION-01.
// Hex literals are the exact binary64 conversions; unnormalized, no Condon--Shortley.
namespace jx::lunar::constants {
inline constexpr double inertia[3] = {0x1.ffad4c4c11ee6p-1, 0x1.ffcb27a08ccefp-1, 0x1.0000000000000p+0};
inline constexpr double polar_scale = 0x1.92714650d870fp-2;
inline constexpr double radius_km = 0x1.b280000000000p+10;
inline constexpr double gm[2] = { 0x1.85421bdf58d8dp+18, 0x1.ee64720e94786p+36 };
inline constexpr double cosine[7][7] = {
  {0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
  {0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
  {0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
  {-0x1.1bdbcde4910c7p-17, 0x1.ddd409d7acd0cp-16, 0x1.44d6f3eebf4f7p-18, 0x1.cb77cc122d9abp-20, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
  {0x1.45a0135837dd9p-17, -0x1.7ed9100c5f8c3p-18, -0x1.ab240a47efb18p-20, -0x1.5a7fcf21109c2p-24, -0x1.108faca8a0d5cp-23, 0x0.0p+0, 0x0.0p+0},
  {-0x1.8e79d3c41a0e9p-21, -0x1.d11579193e198p-21, 0x1.7e3fe104e1b79p-21, 0x1.088e4518a88d9p-26, 0x1.706ad41280b9dp-26, 0x1.072e412ab1d48p-27, 0x0.0p+0},
  {0x1.cdf64ff104347p-17, 0x1.42c62f1c7e6e6p-20, -0x1.25b07f5ae16aap-21, -0x1.276e103038e09p-24, 0x1.630d90c5d177dp-30, 0x1.42a7c71a3b804p-30, -0x1.2bfbde9e89bf8p-30},
};
inline constexpr double sine[7][7] = {
  {0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
  {0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
  {0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
  {0x0.0p+0, 0x1.8b60733664d5dp-18, 0x1.bf413eea643d4p-20, -0x1.09a9c11380a6bp-22, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
  {0x0.0p+0, 0x1.a7d5e8a96842ap-20, -0x1.96c9168343d76p-20, -0x1.af5f12952d96ep-21, 0x1.645295fe542e1p-24, 0x0.0p+0, 0x0.0p+0},
  {0x0.0p+0, -0x1.d96aa7f7a73a1p-19, 0x1.6f657fff02894p-23, 0x1.348da4ee0fb9dp-22, 0x1.217b609dae4dfp-31, -0x1.d21477c5eba84p-28, 0x0.0p+0},
  {0x0.0p+0, -0x1.1285c279bb87ep-19, -0x1.218df6074faebp-22, -0x1.313788c450bb5p-24, -0x1.07e935642a447p-26, -0x1.1ec88935e75f3p-27, 0x1.cf020dd131489p-30},
};
}
