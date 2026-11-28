import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';

// base './' => funcționează pe GitHub Pages / Netlify fără configurări extra
export default defineConfig({ base: './', plugins: [tailwindcss()] });
