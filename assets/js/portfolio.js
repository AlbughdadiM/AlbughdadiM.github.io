(() => {
    const root = document.documentElement;
    const header = document.querySelector('.site-header');
    const themeButton = document.querySelector('[data-theme-toggle]');
    const menuButton = document.querySelector('[data-menu-toggle]');
    const navigation = document.querySelector('[data-navigation]');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const getStoredTheme = () => {
        try {
            return localStorage.getItem('portfolio-theme');
        } catch (error) {
            return null;
        }
    };

    const setStoredTheme = (theme) => {
        try {
            localStorage.setItem('portfolio-theme', theme);
        } catch (error) {
            // The chosen theme still applies for the current visit.
        }
    };

    const updateThemeButton = () => {
        if (!themeButton) return;
        const isLight = root.dataset.theme === 'light';
        themeButton.setAttribute('aria-label', `Switch to ${isLight ? 'dark' : 'light'} theme`);
        themeButton.setAttribute('title', `Switch to ${isLight ? 'dark' : 'light'} theme`);
        const icon = themeButton.querySelector('.theme-icon');
        if (icon) icon.textContent = isLight ? '\u263e' : '\u2600';
    };

    const preferredTheme = getStoredTheme()
        || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    root.dataset.theme = preferredTheme;
    updateThemeButton();

    themeButton?.addEventListener('click', () => {
        const nextTheme = root.dataset.theme === 'light' ? 'dark' : 'light';
        root.dataset.theme = nextTheme;
        setStoredTheme(nextTheme);
        updateThemeButton();
    });

    const closeMenu = () => {
        if (!menuButton || !navigation) return;
        menuButton.setAttribute('aria-expanded', 'false');
        menuButton.setAttribute('aria-label', 'Open navigation');
        navigation.classList.remove('is-open');
    };

    menuButton?.addEventListener('click', () => {
        if (!navigation) return;
        const willOpen = menuButton.getAttribute('aria-expanded') !== 'true';
        menuButton.setAttribute('aria-expanded', String(willOpen));
        menuButton.setAttribute('aria-label', willOpen ? 'Close navigation' : 'Open navigation');
        navigation.classList.toggle('is-open', willOpen);
    });

    navigation?.addEventListener('click', (event) => {
        if (event.target.closest('a')) closeMenu();
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') closeMenu();
    });

    document.addEventListener('click', (event) => {
        if (!navigation?.classList.contains('is-open')) return;
        if (!navigation.contains(event.target) && !menuButton?.contains(event.target)) closeMenu();
    });

    const updateHeader = () => header?.classList.toggle('is-scrolled', window.scrollY > 8);
    updateHeader();
    window.addEventListener('scroll', updateHeader, { passive: true });

    document.querySelectorAll('[data-year]').forEach((year) => {
        year.textContent = new Date().getFullYear();
    });

    const filterButtons = [...document.querySelectorAll('[data-filter]')];
    const projectCards = [...document.querySelectorAll('[data-category]')];
    const filterStatus = document.querySelector('[data-filter-status]');

    filterButtons.forEach((button) => {
        button.addEventListener('click', () => {
            const filter = button.dataset.filter;
            let visibleCount = 0;

            filterButtons.forEach((item) => {
                item.setAttribute('aria-pressed', String(item === button));
            });

            projectCards.forEach((card) => {
                const categories = card.dataset.category.split(' ');
                const isVisible = filter === 'all' || categories.includes(filter);
                card.hidden = !isVisible;
                if (isVisible) visibleCount += 1;
            });

            if (filterStatus) {
                const label = button.textContent.trim();
                filterStatus.textContent = `${visibleCount} ${label.toLowerCase()} project${visibleCount === 1 ? '' : 's'} shown.`;
            }
        });
    });

    const projectVideos = document.querySelectorAll('[data-project-video]');
    const minimumLoopDuration = 3.2;

    const loadVideo = (video) => {
        const source = video.querySelector('source[data-src]');
        if (source) {
            source.src = source.dataset.src;
            source.removeAttribute('data-src');
            video.load();
        }

        const setComfortablePlaybackRate = () => {
            if (!Number.isFinite(video.duration) || video.duration <= 0) return;
            video.playbackRate = Math.min(1, video.duration / minimumLoopDuration);
        };

        if (video.readyState >= 1) {
            setComfortablePlaybackRate();
        } else {
            video.addEventListener('loadedmetadata', setComfortablePlaybackRate, { once: true });
        }
    };

    const playVideo = (video) => {
        loadVideo(video);
        const startPlayback = () => {
            video.play().catch(() => {
                // The still poster remains useful if autoplay is unavailable.
            });
        };

        if (video.readyState >= 2) {
            startPlayback();
        } else {
            video.addEventListener('canplay', startPlayback, { once: true });
        }
    };

    projectVideos.forEach((video) => {
        const toggle = document.createElement('button');
        toggle.className = 'media-toggle';
        toggle.type = 'button';

        const updateToggle = () => {
            const isPaused = video.paused;
            toggle.textContent = isPaused ? '\u25b6 Play' : '\u23f8 Pause';
            toggle.setAttribute('aria-label', `${isPaused ? 'Play' : 'Pause'} ${video.getAttribute('aria-label') || 'project preview'}`);
        };

        toggle.addEventListener('click', () => {
            if (video.paused) {
                video.removeAttribute('data-user-paused');
                playVideo(video);
            } else {
                video.dataset.userPaused = 'true';
                video.pause();
            }
        });

        video.addEventListener('play', updateToggle);
        video.addEventListener('pause', updateToggle);
        updateToggle();
        video.parentElement?.append(toggle);
    });

    const projectGifs = document.querySelectorAll('[data-project-gif]');
    projectGifs.forEach((image) => {
        const animatedSource = image.getAttribute('src');
        const posterSource = image.dataset.poster;
        if (!animatedSource || !posterSource) return;

        const label = image.alt.replace(/^Animated\s+/i, '') || 'project preview';
        const toggle = document.createElement('button');
        toggle.className = 'media-toggle';
        toggle.type = 'button';

        const setPlaying = (shouldPlay) => {
            if (shouldPlay) {
                image.removeAttribute('data-gif-paused');
                if (image.getAttribute('src') !== animatedSource) image.src = animatedSource;
            } else {
                image.dataset.gifPaused = 'true';
                if (image.getAttribute('src') !== posterSource) image.src = posterSource;
            }

            toggle.textContent = shouldPlay ? '\u23f8 Pause' : '\u25b6 Play';
            toggle.setAttribute('aria-label', `${shouldPlay ? 'Pause' : 'Play'} ${label}`);
        };

        toggle.addEventListener('click', () => {
            setPlaying(image.hasAttribute('data-gif-paused'));
        });

        image.parentElement?.append(toggle);
        setPlaying(!reducedMotion);
    });

    if (!reducedMotion && 'IntersectionObserver' in window) {
        const videoObserver = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                const video = entry.target;
                if (entry.isIntersecting) {
                    if (!video.hasAttribute('data-user-paused')) playVideo(video);
                } else if (video.currentSrc) {
                    video.pause();
                }
            });
        }, { rootMargin: '300px 0px', threshold: 0.01 });

        projectVideos.forEach((video) => videoObserver.observe(video));
    }

    const revealItems = document.querySelectorAll('.reveal');
    if (!reducedMotion && 'IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries, currentObserver) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                entry.target.classList.add('is-visible');
                currentObserver.unobserve(entry.target);
            });
        }, { rootMargin: '0px 0px -7% 0px', threshold: 0.08 });

        revealItems.forEach((item) => observer.observe(item));
    } else {
        revealItems.forEach((item) => item.classList.add('is-visible'));
    }
})();
