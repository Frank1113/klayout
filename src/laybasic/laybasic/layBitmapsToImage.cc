
/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

*/


#include "layBitmapsToImage.h"
#include "layBitmap.h"
#include "layDitherPattern.h"
#include "layLineStyles.h"
#include "tlPixelBuffer.h"
#include "tlTimer.h"
#include "tlAssert.h"
#include "tlThreads.h"

#include <cstring>

namespace lay
{

static inline unsigned int
low_bit_index (uint32_t bits)
{
#if defined(__GNUC__) || defined(__clang__)
  return (unsigned int) __builtin_ctz (bits);
#else
  static const unsigned char index[32] = {
    0, 1, 28, 2, 29, 14, 24, 3,
    30, 22, 20, 15, 25, 17, 4, 8,
    31, 27, 13, 23, 21, 19, 16, 7,
    26, 12, 18, 6, 11, 5, 10, 9
  };

  return index [((bits & -bits) * 0x077cb531U) >> 27];
#endif
}

static inline void
fill_color_run (tl::color_t *pt, unsigned int n, tl::color_t c)
{
  while (n-- > 0) {
    *pt++ = c;
  }
}

static inline void
or_color_run (tl::color_t *pt, unsigned int n, tl::color_t c)
{
  while (n-- > 0) {
    *pt++ |= c;
  }
}

static inline void
mask_color_run (tl::color_t *pt, unsigned int n, tl::color_t and_mask, tl::color_t c)
{
  while (n-- > 0) {
    *pt = (*pt & and_mask) | c;
    ++pt;
  }
}

struct ColorPlaneMask
{
  tl::color_t or_mask;
  tl::color_t and_mask;
  tl::color_t color;
  bool copy;
};

struct MonoPlaneMask
{
  bool set_bits;
  bool keep_bits;
};

static void
render_scanline_std (const uint32_t *dp, unsigned int ds, const lay::Bitmap *pbitmap, unsigned int y, unsigned int w, unsigned int /*h*/, uint32_t *data)
{
  const uint32_t *ps = pbitmap->scanline (y);

  if (ds == 1) {
    unsigned int nwords = (w + lay::wordlen - 1) / lay::wordlen;
    if (*dp == lay::wordones) {
      memcpy (data, ps, nwords * sizeof (uint32_t));
      return;
    } else if (*dp == 0) {
      memset (data, 0, nwords * sizeof (uint32_t));
      return;
    }
  }

  if (pbitmap->is_scanline_empty (y)) {
    memset (data, 0, ((w + lay::wordlen - 1) / lay::wordlen) * sizeof (uint32_t));
    return;
  }

  const uint32_t *dm = dp;

  unsigned int x = w;
  while (x >= lay::wordlen) {
    *data++ = *ps++ & *dm++;
    if (dm == dp + ds) {
      dm = dp;
    }
    x -= lay::wordlen;
  }

  if (x > 0) {
    *data = *ps & *dm;
  }
}

static void
render_scanline_std_edge (const uint32_t *dp, unsigned int ds, const lay::Bitmap *pbitmap, unsigned int y, unsigned int w, unsigned int h, uint32_t *data)
{
  if (pbitmap->is_scanline_empty (y)) {
    memset (data, 0, ((w + lay::wordlen - 1) / lay::wordlen) * sizeof (uint32_t));
    return;
  }

  const uint32_t *psp = (y > 0 ? pbitmap->scanline (y - 1) : pbitmap->empty_scanline ());
  const uint32_t *psn = (y < h - 1 ? pbitmap->scanline (y + 1) : pbitmap->empty_scanline ());
  const uint32_t *ps = pbitmap->scanline (y);

  const uint32_t *dm = dp;

  unsigned int b = (y % (32 * ds));
  bool vflag = (dp[b / 32] & (1 << (b % 32))) != 0;

  uint32_t ddp = 0;

  int x = int (w);
  while (x > 0) {

    uint32_t d = 0;
    uint32_t dsn = 0, dsp = 0;
    uint32_t ddn = 0;

    if (x > int (lay::wordlen)) {
      d = *ps++;
      dsn = *psn++;
      dsp = *psp++;
      ddn = *ps;
    } else {
      d = *ps;
      dsn = *psn;
      dsp = *psp;
      if (x < int (lay::wordlen)) {
        d &= ((1 << x) - 1);
      }
    }

    //  di selects the inner bits - such that have a left, right neighbor
    uint32_t dhn1 = (d & ((d >> 1) | ((ddn & 1) << 31)));
    uint32_t dhn2 = (d & ((d << 1) | ((ddp >> 31) & 1)));
    uint32_t dhi = dhn1 & dhn2;
    uint32_t dhn = dhn1 | dhn2;

    //  dvi selects the vertically inner bits - such that have a top and botton neighbor
    uint32_t dvn1 = dsn & d;
    uint32_t dvn2 = dsp & d;
    uint32_t dvi = dvn1 & dvn2;
    uint32_t dvn = dvn1 | dvn2;

#if 1
    /*
      NOTE: this solution is ugly for lines with angles a little away from 45 degree
      like 30..40 and 50..60 degree.
      This is the truth table of the various combinations of bits that are
      encountered. Each combination gets horizontal or vertical bit masks.
      This is basically an edge detection algorithm.
      The diagonal pixels are not considered currently.

      configuration   use mask   dhi   dvi   dhn   dvn
      --------------------------------------------------
       .              H          0     0     0     0
      .x.             [sol] (could be 1 too but covers diagonal edges too,
       .                     so we make it H masked)

       .    .    .    H          x     0     1     x
      xx.  .xx  xxx   [with_hm]
       .    .    .

       x    x    x
      xx.  .xx  xxx
       .    .    .

       .    .    .
      xx.  .xx  xxx
       x    x    x

       x    .    x    V          0     x     x     1
      .x.  .x.  .x.   [with_vm] (the four corner cases will be H and V masks
       .    x    x               and both results get merged)

       x    .    x
      xx.  xx.  xx.
       .    x    x

       x    .    x
      .xx  .xx  .xx
       .    x    x

       x              H*V        1     1     1     1
      xxx             [with_hvm]
       x

     */

    uint32_t sol      = d - (d & (dhi | dvi | dhn | dvn));
    uint32_t with_hm  = (d - (d & dvi)) & dhn;
    uint32_t with_vm  = (d - (d & dhi)) & dvn;
    uint32_t with_hvm = d & dhi & dvi & dhn & dvn;

    uint32_t hm = *dm++;
    uint32_t dd = (sol | with_hm) & hm;
    if (vflag) {
      dd |= with_vm | (with_hvm & hm);
    }

#else
    /*
      NOTE: this alternative solution is ugly for very steep angles
      such as 60 to 85 degree.

      It is based on the following configuration table:

      configuration   use mask   dhi   dvi   dhn   dvn
      --------------------------------------------------
       .              H          0     0     0     0
      .x.             [sol]
       .

       .    .         H          0     0     1     0
      xx.  .xx        [twin]
       .    .

       .    x         H          0     0     0     1
      .x.  .x.        [twin]
       x    .

       .    x         H|V        0     0     1     1
      .xx  .xx        [corner]
       x    .

       .    x
      xx.  xx.
       x    .

       x    x    x    V          0     1     x     (1)
      .x.  .xx  xx.   [vbar]
       x    x    x

       .    .    x    H          1     0     (1)   x
      xxx  xxx  xxx   [hbar]
       .    x    .

       x              H*V        1     1     (1)   (1)
      xxx
       x

     */

    uint32_t sol    = d - (d & (dhi | dvi | dhn | dvn));
    uint32_t twin   = (d - (d & (dhi | dvi))) & (dhn ^ dvn);
    uint32_t corner = (d - (d & (dhi | dvi))) & dhn & dvn;
    uint32_t hbar   = (d - (d & dvi)) & dhi;
    uint32_t vbar   = (d - (d & dhi)) & dvi;
    uint32_t inner  = d & dhi & dvi;

    uint32_t hm = *dm++;
    uint32_t dd = 0;
    //  H (and H|V)
    dd = (sol | twin | hbar | corner /*H|V*/) & hm;
    if (vflag) {
      //  V (and H|V)
      dd |= vbar | corner /*H|V*/;
      //  H*V
      dd |= inner & hm;
    }

#endif

    *data++ = dd;

    if (dm == dp + ds) {
      dm = dp;
    }

    x -= lay::wordlen;
    ddp = d;

  }
}

static void
render_scanline_px (const uint32_t *dp, unsigned int ds, const lay::Bitmap *pbitmap,
                    unsigned int y, unsigned int w, unsigned int h, uint32_t *data,
                    unsigned int pixels)
{
  if (pixels < 1) {
    return;
  }

  if (pixels > 15) {
    pixels = 15;
  }

  const uint32_t *dm = dp;

  unsigned int px1 = (pixels - 1) / 2;
  unsigned int px2 = (pixels - 1) - px1;

  const uint32_t *ps[16];
  bool all_empty = true;
  for (unsigned int p = 0; p < pixels; ++p) {
    unsigned int yy = 0;
    if (y + p < px1) {
      yy = 0;
    } else if ((y + p - px1) >= h) {
      yy = h - 1;
    } else {
      yy = y + p - px1;
    }
    ps[p] = pbitmap->scanline (yy);
    all_empty = all_empty && pbitmap->is_scanline_empty (yy);
  }

  if (all_empty) {
    memset (data, 0, ((w + lay::wordlen - 1) / lay::wordlen) * sizeof (uint32_t));
    return;
  }

  uint32_t d, dd = 0, dn = 0;
  for (unsigned int p = 0; p < pixels; ++p) {
    dn |= *(ps[p]++);
  }

  unsigned int x = w;
  while (true) {

    d = dn;

    dn = 0;
    if (x > lay::wordlen) {
      for (unsigned int p = 0; p < pixels; ++p) {
        dn |= *(ps[p]++);
      }
    }

    uint32_t d0 = d;
    for (unsigned int p = 1; p <= px1; ++p) {
      d |= (d0 >> p) | (dn << (32 - p));
    }
    for (unsigned int p = 1; p <= px2; ++p) {
      d |= (d0 << p) | (dd >> (32 - p));
    }

    dd = d0;

    *data++ = d & *dm++;
    if (dm == dp + ds) {
      dm = dp;
    }

    if (x > lay::wordlen) {
      x -= lay::wordlen;
    } else {
      break;
    }

  }

}

static void
render_scanline_cross (const uint32_t *dp, unsigned int ds, const lay::Bitmap *pbitmap,
                       unsigned int y, unsigned int w, unsigned int h, uint32_t *data,
                       unsigned int pixels)
{
  if (pixels < 1) {
    return;
  }

  //  NOTE: hardcoded bar/width ratio for crosses.
  unsigned int lw = std::max (std::min ((unsigned int) 6, pixels / 9), (unsigned int) 1);

  const int max_pixels = 31;
  if (pixels > max_pixels) {
    pixels = max_pixels;
  }

  const uint32_t *dm = dp;
  unsigned int px1 = (pixels - 1) / 2;
  unsigned int px2 = (pixels - 1) - px1;

  unsigned int spx1 = (lw - 1) / 2;
  unsigned int spx2 = (lw - 1) - spx1;

  const uint32_t *ps[max_pixels + 1];
  bool all_empty = true;
  for (unsigned int p = 0; p < pixels; ++p) {
    unsigned int yy = 0;
    if (y + p < px1) {
      yy = 0;
    } else if ((y + p - px1) >= h) {
      yy = h - 1;
    } else {
      yy = y + p - px1;
    }
    ps[p] = pbitmap->scanline (yy);
    all_empty = all_empty && pbitmap->is_scanline_empty (yy);
  }

  unsigned int nwords = (w + lay::wordlen - 1) / lay::wordlen;
  memset (data, 0, nwords * sizeof (uint32_t));
  if (all_empty) {
    return;
  }

  uint32_t *dpp = data;
  for (unsigned int o = 0; o < pixels; ++o) {

    dpp = data;

    unsigned int bpx1 = 0, bpx2 = 0;
    if (o >= px1 - spx1 && o <= px1 + spx2) {
      bpx1 = px1;
      bpx2 = px2;
    } else {
      bpx1 = spx1;
      bpx2 = spx2;
    }

    if (bpx1 > 0 || bpx2 > 0) {

      uint32_t d, dd = 0, dn;
      dn = *(ps[o]++);

      unsigned int x = w;
      while (true) {

        d = dn;

        dn = 0;
        if (x > lay::wordlen) {
          dn = *(ps[o]++);
        }

        uint32_t d0 = d;
        if (d0 != 0) {
          for (unsigned int p = 1; p <= bpx1; ++p) {
            d |= (d0 >> p);
          }
          for (unsigned int p = 1; p <= bpx2; ++p) {
            d |= (d0 << p);
          }
        }
        if (dn != 0) {
          for (unsigned int p = 1; p <= bpx1; ++p) {
            d |= (dn << (32 - p));
          }
        }
        if (dd != 0) {
          for (unsigned int p = 1; p <= bpx2; ++p) {
            d |= (dd >> (32 - p));
          }
        }

        dd = d0;

        *dpp++ |= d & *dm++;
        if (dm == dp + ds) {
          dm = dp;
        }

        if (x > lay::wordlen) {
          x -= lay::wordlen;
        } else {
          break;
        }

      }

    } else {

      unsigned int x = w;
      while (true) {

        uint32_t d = *(ps[o]++);

        *dpp++ |= d & *dm++;
        if (dm == dp + ds) {
          dm = dp;
        }

        if (x > lay::wordlen) {
          x -= lay::wordlen;
        } else {
          break;
        }

      }

    }

  }
}

static void create_precursor_bitmaps (const std::vector<lay::ViewOp> &view_ops_in, const std::vector <unsigned int> &vo_map, const std::vector<lay::Bitmap *> &pbitmaps_in, const std::vector<unsigned int> &bm_map, const lay::LineStyles &ls, unsigned int width, unsigned int height, std::map<unsigned int, lay::Bitmap> &precursors, tl::Mutex *mutex)
{
  tl_assert (bm_map.size () == vo_map.size ());

  //  Styled lines with width > 1 are not rendered directly, but through an intermediate step.
  //  We prepare the necessary precursor bitmaps now
  for (unsigned int i = 0; i < vo_map.size (); ++i) {

    unsigned int vo_index = vo_map [i];
    unsigned int bm_index = bm_map [i];

    const ViewOp &op = view_ops_in [vo_index];
    if (op.width () > 1 && ls.style (op.line_style_index ()).width () > 0) {

      //  lock bitmaps against change by the redraw thread
      if (mutex) {
        mutex->lock ();
      }

      lay::Bitmap &bp = precursors.insert (std::make_pair (bm_index, lay::Bitmap (width, height, 1.0, 1.0))).first->second;
      const LineStyleInfo &ls_info = ls.style (op.line_style_index ()).scaled (op.width ());

      for (unsigned int y = 0; y < height; y++) {
        render_scanline_std_edge (ls_info.pattern (), ls_info.pattern_stride (), pbitmaps_in [bm_index], y, width, height, bp.scanline (y));
      }

      if (mutex) {
        mutex->unlock ();
      }

    }

  }
}

void
bitmaps_to_image (const std::vector<lay::ViewOp> &view_ops_in,
                  const std::vector<lay::Bitmap *> &pbitmaps_in,
                  const lay::DitherPattern &dp,
                  const lay::LineStyles &ls,
                  double dpr,
                  tl::PixelBuffer *pimage, unsigned int width, unsigned int height,
                  bool use_bitmap_index,
                  tl::Mutex *mutex)
{
  bool transparent = pimage->transparent ();

  std::vector<unsigned int> bm_map;
  std::vector<unsigned int> vo_map;

  vo_map.reserve (view_ops_in.size ());
  bm_map.reserve (view_ops_in.size ());
  unsigned int n_in = 0;

  //  drop invisible and empty bitmaps, build bitmap mask
  for (unsigned int i = 0; i < view_ops_in.size (); ++i) {

    const lay::ViewOp &vop = view_ops_in [i];

    unsigned int bi = (use_bitmap_index && vop.bitmap_index () >= 0) ? (unsigned int) vop.bitmap_index () : i;
    const lay::Bitmap *pb = bi < pbitmaps_in.size () ? pbitmaps_in [bi] : 0;

    if ((vop.ormask () | ~vop.andmask ()) != 0 && pb && ! pb->empty ()) {
      vo_map.push_back (i);
      bm_map.push_back (bi);
      ++n_in;
    }

  }

  //  Styled lines with width > 1 are not rendered directly, but through an intermediate step.
  //  We prepare the necessary precursor bitmaps now
  std::map<unsigned int, lay::Bitmap> precursors;
  create_precursor_bitmaps (view_ops_in, vo_map, pbitmaps_in, bm_map, ls, width, height, precursors, mutex);

  if (n_in == 0) {
    return;
  }

  std::vector<const LineStyleInfo *> line_styles_in;
  std::vector<const DitherPatternInfo *> dither_patterns_in;
  std::vector<std::pair <tl::color_t, tl::color_t> > masks_in;
  line_styles_in.reserve (n_in);
  dither_patterns_in.reserve (n_in);
  masks_in.reserve (n_in);
  for (unsigned int i = 0; i < n_in; ++i) {
    const lay::ViewOp &vop = view_ops_in [vo_map [i]];
    unsigned int line_width = vop.width () > 0 ? (unsigned int) vop.width () : 0;
    line_styles_in.push_back (&ls.style (vop.line_style_index ()).scaled (line_width));
    dither_patterns_in.push_back (&dp.pattern (vop.dither_index ()).scaled (dpr));
    masks_in.push_back (std::make_pair (vop.ormask () & 0x00ffffff,
                                        ~vop.ormask () & vop.andmask () & 0x00ffffff));
  }

  std::vector<lay::ViewOp> view_ops;
  std::vector<const lay::Bitmap *> pbitmaps;
  std::vector<const LineStyleInfo *> line_styles;
  std::vector<const DitherPatternInfo *> dither_patterns;
  std::vector<std::pair <tl::color_t, tl::color_t> > layer_masks;
  std::vector<ColorPlaneMask> masks;
  std::vector<uint32_t> non_empty_sls;

  view_ops.reserve (n_in);
  pbitmaps.reserve (n_in);
  line_styles.reserve (n_in);
  dither_patterns.reserve (n_in);
  layer_masks.reserve (n_in);
  masks.reserve (n_in);
  non_empty_sls.reserve (n_in);

  //  to optimize the bitmap generation, the bitmaps are checked
  //  for emptyness in slices of "slice" scanlines
  unsigned int slice = 32;

  //  allocate a pixel buffer large enough to hold a scanline for all 
  //  planes.
  unsigned int nwords = (width + 31) / 32;
  uint32_t *buffer = new uint32_t [n_in * nwords];

  for (unsigned int y = 0; y < height; y++) {

    //  lock bitmaps against change by the redraw thread
    if (mutex) {
      mutex->lock ();
    }

    //  every "slice" scan lines test what bitmaps are empty 
    if (y % slice == 0) { 

      view_ops.clear ();
      pbitmaps.clear ();
      line_styles.clear ();
      dither_patterns.clear ();
      layer_masks.clear ();
      non_empty_sls.clear ();
      for (unsigned int i = 0; i < n_in; ++i) {

        const lay::ViewOp &vop = view_ops_in [vo_map[i]];
        unsigned int w = vop.width ();

        const lay::Bitmap *pb = 0;
        unsigned int bm_index = bm_map[i];
        if (bm_map [i] < pbitmaps_in.size ()) {
          if (w > 1 && ls.style (vop.line_style_index ()).width () > 0) {
            tl_assert (precursors.find (bm_index) != precursors.end ());
            pb = &precursors [bm_index];
          } else {
            pb = pbitmaps_in [bm_index];
          }
        }

        if (pb != 0 
            && w > 0
            && ((pb->first_scanline () < y + slice && pb->last_scanline () > y) || w > 1)
            && (vop.ormask () | ~vop.andmask ()) != 0) {

          uint32_t non_empty_sl = 0;
          uint32_t m = 1;

          for (unsigned int yy = 0; yy < slice && yy + y < height; ++yy, m <<= 1) {
            if (! pb->is_scanline_empty (yy + y)) {
              non_empty_sl |= m;
            }
          }

          if (non_empty_sl || w > 1) {
            view_ops.push_back (vop);
            pbitmaps.push_back (pb);
            line_styles.push_back (line_styles_in [i]);
            dither_patterns.push_back (dither_patterns_in [i]);
            layer_masks.push_back (masks_in [i]);
            non_empty_sls.push_back (non_empty_sl);
          }

        }

      }

    } 

    //  Collect all necessary information to transfer a single scanline ..
    
    masks.clear ();

    const uint32_t fill_bits   = 0xff000000; // fill alpha value with ones
    uint32_t *dptr = buffer;
    uint32_t ne_mask = (1 << (y % slice));
    for (unsigned int i = 0; i < view_ops.size (); ++i) {

      const ViewOp &op = view_ops [i];
      if (op.width () > 1 || (op.width () == 1 && (non_empty_sls [i] & ne_mask) != 0)) {

        const LineStyleInfo &ls_info = *line_styles [i];
        const DitherPatternInfo &dp_info = *dither_patterns [i];
        const uint32_t *dither = dp_info.pattern () [(y + op.dither_offset ()) % dp_info.height ()];
        if (dither != 0) {

          unsigned int dither_stride = dp_info.pattern_stride ();

          ColorPlaneMask mask;
          mask.or_mask = layer_masks [i].first;
          mask.and_mask = layer_masks [i].second;
          mask.color = mask.or_mask | fill_bits;
          mask.copy = (mask.and_mask == 0);
          masks.push_back (mask);

          if (op.width () == 1) {
            if (ls_info.width () > 0) {
              render_scanline_std_edge (ls_info.pattern (), ls_info.pattern_stride (), pbitmaps [i], y, width, height, dptr);
            } else {
              render_scanline_std (dither, dither_stride, pbitmaps [i], y, width, height, dptr);
            }
          } else if (op.width () > 1) {
            if (op.shape () == lay::ViewOp::Rect) {
              render_scanline_px (dither, dither_stride, pbitmaps [i], y, width, height, dptr, (unsigned int) op.width ());
            } else if (op.shape () == lay::ViewOp::Cross) {
              render_scanline_cross (dither, dither_stride, pbitmaps [i], y, width, height, dptr, (unsigned int) op.width ());
            }
          }

          dptr += nwords;

        }

      }

    }

    //  unlock bitmaps against change by the redraw thread
    if (mutex) {
      mutex->unlock ();
    }

    //  .. and do the actual transfer.

    if (masks.size () > 0) {

      tl::color_t *pt = (tl::color_t *) pimage->scan_line (height - 1 - y);
      uint32_t *dptr_end = dptr; 
      const unsigned int full_words = width / 32;
      const unsigned int tail_pixels = width & 31;

      if (masks.size () == 1) {

        const uint32_t and_mask = masks [0].and_mask;
        const uint32_t color = masks [0].color;
        dptr = dptr_end - nwords;

        for (unsigned int i = 0; i < full_words; ++i, ++dptr, pt += 32) {
          uint32_t d = *dptr;
          if (and_mask == 0 && d == lay::wordones) {
            fill_color_run (pt, 32, color);
          } else if (d == lay::wordones) {
            mask_color_run (pt, 32, and_mask, color);
          } else {
            if (! transparent) {
              or_color_run (pt, 32, fill_bits);
            }
            while (d != 0) {
              unsigned int k = low_bit_index (d);
              pt [k] = (pt [k] & and_mask) | color;
              d &= d - 1;
            }
          }
        }

        if (tail_pixels > 0) {
          const uint32_t word_mask = (uint32_t (1) << tail_pixels) - 1;
          uint32_t d = *dptr & word_mask;

          if (and_mask == 0 && d == word_mask) {
            fill_color_run (pt, tail_pixels, color);
          } else if (d == word_mask) {
            mask_color_run (pt, tail_pixels, and_mask, color);
          } else {
            if (! transparent) {
              or_color_run (pt, tail_pixels, fill_bits);
            }
            while (d != 0) {
              unsigned int k = low_bit_index (d);
              pt [k] = (pt [k] & and_mask) | color;
              d &= d - 1;
            }
          }
        }

      } else {

        bool all_copy = true;
        for (std::vector<ColorPlaneMask>::const_iterator m = masks.begin (); m != masks.end (); ++m) {
          if (! m->copy) {
            all_copy = false;
            break;
          }
        }

        for (unsigned int i = 0; i < full_words; ++i) {

          const unsigned int n = 32;
          const uint32_t word_mask = lay::wordones;

          if (all_copy) {

            uint32_t remaining = word_mask;
            dptr = dptr_end - nwords + i;
            for (int j = int (masks.size () - 1); j >= 0 && remaining != 0; --j) {
              uint32_t d = *dptr & remaining;
              if (d != 0) {
                const tl::color_t c = masks [j].color;
                if (remaining == word_mask && d == word_mask) {
                  fill_color_run (pt, n, c);
                  remaining = 0;
                  break;
                } else if (d == remaining) {
                  uint32_t r = remaining;
                  while (r != 0) {
                    unsigned int k = low_bit_index (r);
                    pt [k] = c;
                    r &= r - 1;
                  }
                  remaining = 0;
                  break;
                }
                while (d != 0) {
                  unsigned int k = low_bit_index (d);
                  pt [k] = c;
                  d &= d - 1;
                }
                remaining &= ~*dptr;
              }
              dptr -= nwords;
            }

            if (! transparent) {
              if (remaining == word_mask) {
                or_color_run (pt, n, fill_bits);
              } else {
                while (remaining != 0) {
                  unsigned int k = low_bit_index (remaining);
                  pt [k] |= fill_bits;
                  remaining &= remaining - 1;
                }
              }
            }

            pt += n;

          } else {

            bool initialized = false;
            tl::color_t y[32];
            tl::color_t z[32];

            dptr = dptr_end - nwords + i;
            if (transparent) {

              for (int j = int (masks.size () - 1); j >= 0; --j) {

                uint32_t d = *dptr & word_mask;
                if (d != 0) {
                  if (! initialized) {
                    memset (y, 0, sizeof (y));
                    memset (z, 0xff, sizeof (z));
                    initialized = true;
                  }
                  const uint32_t or_mask = masks [j].or_mask;
                  const uint32_t and_mask = masks [j].and_mask;
                  while (d != 0) {
                    unsigned int k = low_bit_index (d);
                    y [k] |= (or_mask & z [k]) | fill_bits;
                    z [k] &= and_mask;
                    d &= d - 1;
                  }
                }

                dptr -= nwords;

              }

            } else {

              for (int j = int (masks.size () - 1); j >= 0; --j) {

                uint32_t d = *dptr & word_mask;
                if (d != 0) {
                  if (! initialized) {
                    fill_color_run (y, 32, fill_bits);
                    memset (z, 0xff, sizeof (z));
                    initialized = true;
                  }
                  const uint32_t or_mask = masks [j].or_mask;
                  const uint32_t and_mask = masks [j].and_mask;
                  while (d != 0) {
                    unsigned int k = low_bit_index (d);
                    y [k] |= or_mask & z [k];
                    z [k] &= and_mask;
                    d &= d - 1;
                  }
                }

                dptr -= nwords;

              }
            }

            if (initialized) {
              for (unsigned int k = 0; k < n; ++k) {
                *pt = (*pt & z[k]) | y[k];
                ++pt;
              }
            } else {
              if (! transparent) {
                or_color_run (pt, n, fill_bits);
              }
              pt += n;
            }

          }
        }

        if (tail_pixels > 0) {

          const unsigned int i = full_words;
          const unsigned int n = tail_pixels;
          const uint32_t word_mask = (uint32_t (1) << tail_pixels) - 1;

          if (all_copy) {

            uint32_t remaining = word_mask;
            dptr = dptr_end - nwords + i;
            for (int j = int (masks.size () - 1); j >= 0 && remaining != 0; --j) {
              uint32_t d = *dptr & remaining;
              if (d != 0) {
                const tl::color_t c = masks [j].color;
                if (remaining == word_mask && d == word_mask) {
                  fill_color_run (pt, n, c);
                  remaining = 0;
                  break;
                } else if (d == remaining) {
                  uint32_t r = remaining;
                  while (r != 0) {
                    unsigned int k = low_bit_index (r);
                    pt [k] = c;
                    r &= r - 1;
                  }
                  remaining = 0;
                  break;
                }
                while (d != 0) {
                  unsigned int k = low_bit_index (d);
                  pt [k] = c;
                  d &= d - 1;
                }
                remaining &= ~*dptr;
              }
              dptr -= nwords;
            }

            if (! transparent) {
              if (remaining == word_mask) {
                or_color_run (pt, n, fill_bits);
              } else {
                while (remaining != 0) {
                  unsigned int k = low_bit_index (remaining);
                  pt [k] |= fill_bits;
                  remaining &= remaining - 1;
                }
              }
            }

          } else {

            bool initialized = false;
            tl::color_t y[32];
            tl::color_t z[32];

            dptr = dptr_end - nwords + i;
            if (transparent) {

              for (int j = int (masks.size () - 1); j >= 0; --j) {

                uint32_t d = *dptr & word_mask;
                if (d != 0) {
                  if (! initialized) {
                    memset (y, 0, sizeof (y));
                    memset (z, 0xff, sizeof (z));
                    initialized = true;
                  }
                  const uint32_t or_mask = masks [j].or_mask;
                  const uint32_t and_mask = masks [j].and_mask;
                  while (d != 0) {
                    unsigned int k = low_bit_index (d);
                    y [k] |= (or_mask & z [k]) | fill_bits;
                    z [k] &= and_mask;
                    d &= d - 1;
                  }
                }

                dptr -= nwords;

              }

            } else {

              for (int j = int (masks.size () - 1); j >= 0; --j) {

                uint32_t d = *dptr & word_mask;
                if (d != 0) {
                  if (! initialized) {
                    fill_color_run (y, 32, fill_bits);
                    memset (z, 0xff, sizeof (z));
                    initialized = true;
                  }
                  const uint32_t or_mask = masks [j].or_mask;
                  const uint32_t and_mask = masks [j].and_mask;
                  while (d != 0) {
                    unsigned int k = low_bit_index (d);
                    y [k] |= or_mask & z [k];
                    z [k] &= and_mask;
                    d &= d - 1;
                  }
                }

                dptr -= nwords;

              }
            }

            if (initialized) {
              for (unsigned int k = 0; k < n; ++k) {
                *pt = (*pt & z[k]) | y[k];
                ++pt;
              }
            } else {
              if (! transparent) {
                or_color_run (pt, n, fill_bits);
              }
              pt += n;
            }

          }
        }

      }

    }

  }

  //  free the pixel buffer
  delete [] buffer;
}

void
bitmaps_to_image (const std::vector<lay::ViewOp> &view_ops_in,
                  const std::vector<lay::Bitmap *> &pbitmaps_in,
                  const lay::DitherPattern &dp,
                  const lay::LineStyles &ls,
                  double dpr,
                  tl::BitmapBuffer *pimage, unsigned int width, unsigned int height,
                  bool use_bitmap_index,
                  tl::Mutex *mutex)
{
  std::vector<unsigned int> bm_map;
  std::vector<unsigned int> vo_map;

  vo_map.reserve (view_ops_in.size ());
  bm_map.reserve (view_ops_in.size ());
  unsigned int n_in = 0;

  //  drop invisible and empty bitmaps, build bitmap mask
  for (unsigned int i = 0; i < view_ops_in.size (); ++i) {

    const lay::ViewOp &vop = view_ops_in [i];

    unsigned int bi = (use_bitmap_index && vop.bitmap_index () >= 0) ? (unsigned int) vop.bitmap_index () : i;
    const lay::Bitmap *pb = bi < pbitmaps_in.size () ? pbitmaps_in [bi] : 0;

    if ((vop.ormask () | ~vop.andmask ()) != 0 && pb && ! pb->empty ()) {
      vo_map.push_back (i);
      bm_map.push_back (bi);
      ++n_in;
    }

  }

  //  Styled lines with width > 1 are not rendered directly, but through an intermediate step.
  //  We prepare the necessary precursor bitmaps now
  std::map<unsigned int, lay::Bitmap> precursors;
  create_precursor_bitmaps (view_ops_in, vo_map, pbitmaps_in, bm_map, ls, width, height, precursors, mutex);

  if (n_in == 0) {
    return;
  }

  std::vector<const LineStyleInfo *> line_styles_in;
  std::vector<const DitherPatternInfo *> dither_patterns_in;
  std::vector<std::pair <tl::color_t, tl::color_t> > masks_in;
  line_styles_in.reserve (n_in);
  dither_patterns_in.reserve (n_in);
  masks_in.reserve (n_in);
  for (unsigned int i = 0; i < n_in; ++i) {
    const lay::ViewOp &vop = view_ops_in [vo_map [i]];
    unsigned int line_width = vop.width () > 0 ? (unsigned int) vop.width () : 0;
    line_styles_in.push_back (&ls.style (vop.line_style_index ()).scaled (line_width));
    dither_patterns_in.push_back (&dp.pattern (vop.dither_index ()).scaled (dpr));
    masks_in.push_back (std::make_pair (vop.ormask () & 0x008000,
                                        ~vop.ormask () & vop.andmask () & 0x008000));
  }

  std::vector<lay::ViewOp> view_ops;
  std::vector<const lay::Bitmap *> pbitmaps;
  std::vector<const LineStyleInfo *> line_styles;
  std::vector<const DitherPatternInfo *> dither_patterns;
  std::vector<std::pair <tl::color_t, tl::color_t> > layer_masks;
  std::vector<MonoPlaneMask> masks;
  std::vector<uint32_t> non_empty_sls;

  view_ops.reserve (n_in);
  pbitmaps.reserve (n_in);
  line_styles.reserve (n_in);
  dither_patterns.reserve (n_in);
  layer_masks.reserve (n_in);
  masks.reserve (n_in);
  non_empty_sls.reserve (n_in);

  //  to optimize the bitmap generation, the bitmaps are checked
  //  for emptyness in slices of "slice" scanlines
  unsigned int slice = 32;

  //  allocate a pixel buffer large enough to hold a scanline for all 
  //  planes.
  unsigned int nwords = (width + 31) / 32;
  uint32_t *buffer = new uint32_t [n_in * nwords];

  for (unsigned int y = 0; y < height; y++) {

    //  lock bitmaps against change by the redraw thread
    if (mutex) {
      mutex->lock ();
    }

    //  every "slice" scan lines test what bitmaps are empty 
    if (y % slice == 0) { 

      view_ops.clear ();
      pbitmaps.clear ();
      line_styles.clear ();
      dither_patterns.clear ();
      layer_masks.clear ();
      non_empty_sls.clear ();
      for (unsigned int i = 0; i < n_in; ++i) {

        const lay::ViewOp &vop = view_ops_in [vo_map[i]];
        unsigned int w = vop.width ();

        const lay::Bitmap *pb = 0;
        unsigned int bm_index = bm_map[i];
        if (bm_map [i] < pbitmaps_in.size ()) {
          if (w > 1 && ls.style (vop.line_style_index ()).width () > 0) {
            tl_assert (precursors.find (bm_index) != precursors.end ());
            pb = &precursors [bm_index];
          } else {
            pb = pbitmaps_in [bm_index];
          }
        }

        if (pb != 0
            && w > 0
            && ((pb->first_scanline () < y + slice && pb->last_scanline () > y) || w > 1)
            && (vop.ormask () | ~vop.andmask ()) != 0) {

          uint32_t non_empty_sl = 0;
          uint32_t m = 1;

          for (unsigned int yy = 0; yy < slice && yy + y < height; ++yy, m <<= 1) {
            if (! pb->is_scanline_empty (yy + y)) {
              non_empty_sl |= m;
            }
          }

          if (non_empty_sl || w > 1) {
            view_ops.push_back (vop);
            pbitmaps.push_back (pb);
            line_styles.push_back (line_styles_in [i]);
            dither_patterns.push_back (dither_patterns_in [i]);
            layer_masks.push_back (masks_in [i]);
            non_empty_sls.push_back (non_empty_sl);
          }

        }

      }

    } 

    //  Collect all necessary information to transfer a single scanline ..
    
    masks.clear ();

    uint32_t needed_bits = 0x008000; // only green bit 7 required
    uint32_t *dptr = buffer;
    uint32_t ne_mask = (1 << (y % slice));
    for (unsigned int i = 0; i < view_ops.size (); ++i) {

      const ViewOp &op = view_ops [i];
      if (op.width () > 1 || (op.width () == 1 && (non_empty_sls [i] & ne_mask) != 0)) {

        const LineStyleInfo &ls_info = *line_styles [i];
        const DitherPatternInfo &dp_info = *dither_patterns [i];
        const uint32_t *dither = dp_info.pattern () [(y + op.dither_offset ()) % dp_info.height ()];
        if (dither != 0) {

          unsigned int dither_stride = dp_info.pattern_stride ();

          MonoPlaneMask mask;
          mask.set_bits = (layer_masks [i].first & needed_bits) != 0;
          mask.keep_bits = (layer_masks [i].second & needed_bits) != 0;
          masks.push_back (mask);

          if (op.width () == 1) {
            if (ls_info.width () > 0) {
              render_scanline_std_edge (ls_info.pattern (), ls_info.pattern_stride (), pbitmaps [i], y, width, height, dptr);
            } else {
              render_scanline_std (dither, dither_stride, pbitmaps [i], y, width, height, dptr);
            }
          } else if (op.width () > 1) {
            if (op.shape () == lay::ViewOp::Rect) {
              render_scanline_px (dither, dither_stride, pbitmaps [i], y, width, height, dptr, (unsigned int) op.width ());
            } else if (op.shape () == lay::ViewOp::Cross) {
              render_scanline_cross (dither, dither_stride, pbitmaps [i], y, width, height, dptr, (unsigned int) op.width ());
            }
          }

          dptr += nwords;

        }

      }

    }

    //  unlock bitmaps against change by the redraw thread
    if (mutex) {
      mutex->unlock ();
    }

    //  .. and do the actual transfer.

    if (masks.size () > 0) {

      tl::color_t *pt = (tl::color_t *) pimage->scan_line (height - 1 - y);
      uint32_t *dptr_end = dptr; 

      if (masks.size () == 1) {

        const bool set_bits = masks [0].set_bits;
        const bool keep_bits = masks [0].keep_bits;
        dptr = dptr_end - nwords;

        for (unsigned int x = 0; x < width; x += 32, ++dptr) {
          uint32_t d = *dptr;
          if (width - x < 32) {
            d &= (uint32_t (1) << (width - x)) - 1;
          }
          if (d != 0) {
            if (set_bits) {
              *pt |= d;
            } else if (! keep_bits) {
              *pt &= ~d;
            }
          }
          ++pt;
        }

      } else {

        unsigned int i = 0;
        for (unsigned int x = 0; x < width; x += 32, ++i) {

          uint32_t y = 0;
          uint32_t z = lay::wordones;
          uint32_t word_mask = lay::wordones;
          if (width - x < 32) {
            word_mask = (uint32_t (1) << (width - x)) - 1;
          }

          dptr = dptr_end - nwords + i;
          for (int j = int (masks.size () - 1); j >= 0; --j) {
            uint32_t d = *dptr & word_mask;
            if (d != 0) {
              if (masks [j].set_bits) {
                y |= (z & d);
              }
              if (! masks [j].keep_bits) {
                z &= ~d;
              }
            }
            dptr -= nwords;
          }

          *pt = (*pt & z) | y;
          ++pt;

        }

      }

    }

  }

  //  free the pixel buffer
  delete [] buffer;
}

void
bitmap_to_bitmap (const lay::ViewOp &view_op, const lay::Bitmap &bitmap,
                  unsigned char *data,
                  unsigned int width, unsigned int height,
                  const lay::DitherPattern &dp,
                  const lay::LineStyles &ls,
                  double dpr)
{
  //  quick exit, if line width is zero
  if (view_op.width () == 0) {
    return;
  }

  unsigned int nwords = (width + 31) / 32;
  uint32_t *buffer = new uint32_t [nwords];

  //  determine endianess ..
  unsigned int x = 0xc0000001;
  unsigned char x0 = ((unsigned char *) &x) [0];

  const DitherPatternInfo &dp_info = dp.pattern (view_op.dither_index ()).scaled (dpr);
  const LineStyleInfo &ls_info = ls.style (view_op.line_style_index ()).scaled (view_op.width ());

  for (unsigned int y = 0; y < height; ++y) {

    unsigned int nbytes = ((width + 7) / 8);

    if (view_op.width () > 1 || ! bitmap.is_scanline_empty (height - 1 - y)) {

      const uint32_t *dither = dp_info.pattern () [(height - 1 - y + view_op.dither_offset ()) % dp_info.height ()];
      unsigned int dither_stride = dp_info.pattern_stride ();

      if (view_op.width () == 1) {

        if (ls_info.width () > 0) {
          render_scanline_std_edge (ls_info.pattern (), ls_info.pattern_stride (), &bitmap, height - 1 - y, width, height, buffer);
        } else {
          render_scanline_std (dither, dither_stride, &bitmap, height - 1 - y, width, height, buffer);
        }

      } else if (view_op.width () > 1) {

        const lay::Bitmap *bp = &bitmap;

        //  Styled lines with width > 1 are not rendered directly, but through an intermediate step.
        //  We prepare the necessary precursor bitmaps now
        lay::Bitmap precursor;
        if (ls_info.width () > 0) {

          precursor = lay::Bitmap (width, height, 1.0, 1.0);

          LineStyleInfo lsi = ls_info;

          for (unsigned int y = 0; y < height; y++) {
            render_scanline_std_edge (lsi.pattern (), lsi.pattern_stride (), bp, y, width, height, precursor.scanline (y));
          }

          bp = &precursor;

        }

        if (view_op.shape () == lay::ViewOp::Rect) {
          render_scanline_px (dither, dither_stride, bp, height - 1 - y, width, height, buffer, (unsigned int) view_op.width ());
        } else if (view_op.shape () == lay::ViewOp::Cross) {
          render_scanline_cross (dither, dither_stride, bp, height - 1 - y, width, height, buffer, (unsigned int) view_op.width ());
        }
      }

      const uint32_t *p = buffer;
      uint32_t d = 0;
      const char *dp = (const char *)&d;

      if (x0 == 0xc0) {

        //  MSB first ..
        while (nbytes >= 4) {
          d = *p++;
          if (d) {
            *data++ |= dp[3];
            *data++ |= dp[2];
            *data++ |= dp[1];
            *data++ |= dp[0];
          } else {
            data += 4;
          }
          nbytes -= 4;
        }
        if (nbytes > 0) {
          d = *p++;
          if (d) {
            dp += 4;
            while (nbytes > 0) {
              *data++ |= *--dp;
              --nbytes;
            }
          } else {
            data += nbytes;
          }
        }

      } else if (x0 == 0x01) {

        //  LSB first ..
        while (nbytes >= 4) {
          d = *p++;
          if (d) {
            *data++ |= dp[0];
            *data++ |= dp[1];
            *data++ |= dp[2];
            *data++ |= dp[3];
          } else {
            data += 4;
          }
          nbytes -= 4;
        }
        if (nbytes > 0) {
          d = *p++;
          if (d) {
            while (nbytes > 0) {
              *data++ |= *dp++;
              --nbytes;
            }
          } else {
            data += nbytes;
          }
        }

      } else {
        //  unable to determine endianess
        tl_assert (false);
      }

    } else {
      data += nbytes;
    }
    
  }

  delete [] buffer;
}

}
