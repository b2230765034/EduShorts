// utils/colors.ts
// hex (#fff / #ffffff) ve rgba(...) destekli basit bir kontrast seçici

function parseColor(c: string): { r: number; g: number; b: number } {
    if (!c) return { r: 0, g: 0, b: 0 };
  
    // rgba(r,g,b,a)
    const rgba = c.match(/rgba?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/i);
    if (rgba) return { r: +rgba[1], g: +rgba[2], b: +rgba[3] };
  
    // #rgb veya #rrggbb
    let hex = c.replace('#', '');
    if (hex.length === 3) hex = hex.split('').map(x => x + x).join('');
    const num = parseInt(hex, 16);
    return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
  }
  
  function luminance({ r, g, b }: { r: number; g: number; b: number }) {
    // WCAG relative luminance
    const srgb = [r, g, b].map(v => {
      const x = v / 255;
      return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2];
  }
  
  /** Arka plan rengine göre '#000' veya '#fff' döndürür */
  export function pickReadable(bgColor: string, light = '#fff', dark = '#000') {
    const L = luminance(parseColor(bgColor));
    return L > 0.5 ? dark : light; // arka plan açık ise siyah, koyu ise beyaz
  }
  