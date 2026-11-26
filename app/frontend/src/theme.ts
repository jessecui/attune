// The light/dark choice: a class on <html>, remembered between visits. index.html applies it
// before first paint using the same key.
const KEY = 'attune:theme'

export const isDark = () => document.documentElement.classList.contains('dark')

export function setDark(dark: boolean) {
  document.documentElement.classList.toggle('dark', dark)
  try { localStorage.setItem(KEY, dark ? 'dark' : 'light') } catch { /* private mode: the choice lasts for this visit only */ }
}
