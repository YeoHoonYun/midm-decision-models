// Public-site tags only. Never load these services in the local trading app.
(() => {
  const HOST = 'yeohoonyun.github.io';
  const PREFIX = '/midm-decision-models/';
  const PAGE = `${PREFIX}model-analysis/index.html`;
  const ID = 'G-TZJ6QQCSRR';
  const CLIENT = 'ca-pub-9061090682937413';
  if (location.protocol !== 'https:' || location.hostname !== HOST ||
      ![PAGE, `${PREFIX}model-analysis/`].includes(location.pathname) ||
      location.search || !['', '#scenarios', '#models', '#industries', '#activity'].includes(location.hash) ||
      window.top !== window.self || navigator.doNotTrack === '1' ||
      window.doNotTrack === '1' || navigator.globalPrivacyControl === true ||
      document.documentElement.dataset.publicTags) return;
  document.documentElement.dataset.publicTags = 'initialized';
  window.dataLayer = window.dataLayer || [];
  window.gtag = window.gtag || function () { window.dataLayer.push(arguments); };
  const gtag = window.gtag;
  gtag('consent', 'default', {
    ad_storage: 'denied', ad_user_data: 'denied', ad_personalization: 'denied'
  });
  gtag('set', 'ads_data_redaction', true);
  const page = {
    page_location: `https://${HOST}${PAGE}`,
    page_referrer: '', page_title: 'ScenarioView / model-analysis'
  };
  // Static page metadata also applies to automatically collected engagement.
  // No filter, search text, symbol, account or snapshot content is sent as an event.
  gtag('set', page);
  gtag('js', new Date());
  gtag('config', ID, {
    ...page, send_page_view: false, ignore_referrer: true,
    allow_google_signals: false, allow_ad_personalization_signals: false,
    cookie_path: PREFIX
  });
  gtag('event', 'page_view', { ...page, send_to: ID });
  function load(src, status) {
    const script = document.createElement('script');
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.referrerPolicy = 'no-referrer';
    script.src = src;
    script.addEventListener('load', () => { document.documentElement.dataset[status] = 'loaded'; });
    script.addEventListener('error', () => { document.documentElement.dataset[status] = 'blocked-or-failed'; });
    document.head.append(script);
  }
  load(`https://www.googletagmanager.com/gtag/js?id=${ID}`, 'analyticsScript');
  load(`https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${CLIENT}`, 'adsenseScript');
})();
