(() => {
  const SPOTIFY_SRC = 'https://open.spotify.com/embed/playlist/2R1CuTFpxQvKNrXp9uGTfX?utm_source=generator&theme=0&si=15728c7c1c9b4c8a';
  const PLAYER_ID = 'persistent-spotify';
  const THEME_KEY = 'dark';

  const STATE_KEY = '__konradSiteShell';
  const shell = window[STATE_KEY] || (window[STATE_KEY] = { cachedPlayer: null, navigating: false, bound: false });

  function desiredDark() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored != null) return stored === 'true';
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? true;
  }

  function setTheme(isDark) {
    document.documentElement.classList.toggle('dark', isDark);
    document.body?.classList.toggle('theme-day', !isDark);
    document.body?.classList.toggle('theme-night', isDark);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', isDark ? '#08090b' : '#f4efe4');
    localStorage.setItem(THEME_KEY, String(isDark));
    document.querySelectorAll('.mode-toggle').forEach((button) => {
      button.setAttribute('aria-pressed', String(!isDark));
      button.setAttribute('aria-label', isDark ? 'switch to day mode' : 'switch to night mode');
      button.textContent = isDark ? '☀︎' : '☾';
    });
  }

  function createPlayer() {
    const player = document.createElement('section');
    const iframe = document.createElement('iframe');

    player.id = PLAYER_ID;
    player.className = 'persistent-spotify';
    player.setAttribute('aria-label', 'Spotify playlist');

    iframe.title = 'Spotify playlist';
    iframe.src = SPOTIFY_SRC;
    iframe.width = '100%';
    iframe.height = '80';
    iframe.allow = 'autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture';
    iframe.loading = 'lazy';

    player.appendChild(iframe);
    return player;
  }

  function ensurePlayer() {
    const mounted = document.getElementById(PLAYER_ID);

    if (shell.cachedPlayer) {
      if (mounted && mounted !== shell.cachedPlayer) mounted.remove();
      return shell.cachedPlayer;
    }

    shell.cachedPlayer = mounted || createPlayer();
    return shell.cachedPlayer;
  }

  function mountPlayer() {
    const player = ensurePlayer();
    if (!document.body.classList.contains('landing-page')) {
      player.remove();
      return;
    }

    const portal = document.querySelector('.spotify-portal');
    if (portal && player.parentElement !== portal) {
      portal.appendChild(player);
    } else if (!portal && player.parentElement !== document.body) {
      document.body.appendChild(player);
    } else if (!player.parentElement) {
      document.body.appendChild(player);
    }
  }

  function syncControls() {
    setTheme(desiredDark());
    mountPlayer();
  }

  function renderMath(attempt, generation) {
    if (generation !== shell.mathGeneration) return;

    const content = document.querySelector('.blog-content');
    if (!content) return;

    if (typeof window.renderMathInElement === 'function') {
      window.renderMathInElement(content, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
        ],
        throwOnError: false,
      });
      return;
    }

    // KaTeX may still be downloading after an in-site navigation. Keep this
    // page's render request alive, but abandon it as soon as we navigate again.
    if (attempt < 200) {
      window.setTimeout(() => renderMath(attempt + 1, generation), 50);
    }
  }

  function syncPage() {
    syncControls();
    shell.mathGeneration = (shell.mathGeneration || 0) + 1;
    renderMath(0, shell.mathGeneration);
  }

  function mergeHead(nextDoc) {
    document.title = nextDoc.title;

    const existing = new Set(
      [...document.head.querySelectorAll('link[rel~="stylesheet"], script[src]')].map((el) => el.href || el.src)
    );

    nextDoc.head.querySelectorAll('link[rel~="stylesheet"], script[src]').forEach((node) => {
      const url = node.href || node.src;
      if (!url || existing.has(url) || /site-shell\.js/.test(url)) return;
      const clone = document.createElement(node.tagName.toLowerCase());
      [...node.attributes].forEach((attr) => clone.setAttribute(attr.name, attr.value));
      document.head.appendChild(clone);
      existing.add(url);
    });

    const nextCanonical = nextDoc.head.querySelector('link[rel="canonical"]');
    const canonical = document.head.querySelector('link[rel="canonical"]');
    if (nextCanonical && canonical) canonical.href = nextCanonical.href;
  }

  async function visit(url, push = true) {
    if (shell.navigating) return;
    shell.navigating = true;

    const player = ensurePlayer();
    player.remove();

    try {
      const response = await fetch(url, { headers: { 'X-Site-Shell': '1' } });
      const type = response.headers.get('content-type') || '';
      if (!response.ok || !type.includes('text/html')) {
        window.location.href = url;
        return;
      }

      const html = await response.text();
      const nextDoc = new DOMParser().parseFromString(html, 'text/html');
      nextDoc.getElementById(PLAYER_ID)?.remove();
      mergeHead(nextDoc);

      document.body.className = nextDoc.body.className;
      document.body.innerHTML = nextDoc.body.innerHTML;
      if (push) history.pushState({ siteShell: true }, '', url);
      syncPage();
      window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
    } catch (_) {
      window.location.href = url;
    } finally {
      shell.navigating = false;
    }
  }

  function handleClick(event) {
    const mode = event.target.closest?.('.mode-toggle, .btn-dark');
    if (mode) {
      event.preventDefault();
      setTheme(localStorage.getItem(THEME_KEY) !== 'true');
      return;
    }

    const menu = event.target.closest?.('.btn-menu');
    if (menu) {
      event.preventDefault();
      document.documentElement.classList.toggle('open');
      return;
    }

    const link = event.target.closest?.('a[href]');
    if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (link.target || link.hasAttribute('download')) return;

    const url = new URL(link.href, window.location.href);
    if (url.origin !== window.location.origin) return;
    if (url.pathname === window.location.pathname && url.search === window.location.search && url.hash) return;

    event.preventDefault();
    visit(url.href);
  }

  function handlePopState() {
    visit(window.location.href, false);
  }

  if (!shell.bound) {
    document.addEventListener('click', handleClick);
    window.addEventListener('popstate', handlePopState);
    shell.bound = true;
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', syncPage, { once: true });
  } else {
    syncPage();
  }
})();
