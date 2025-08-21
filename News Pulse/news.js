// news.js – Fully Updated with Rate Limiting, Caching, Loading States, Modal Previews, and API Monitoring
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = "https://buganeskvrtcnzdumjzo.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFub24iLCJpYXQiOjE3NTU3NTU1MjQsImV4cCI6MjA3MTMzMTUyNH0.vFRUBLKs71S7TtnLm6jNsRNVI-FQtA2SeoJNdkjZsOk";
const sb = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// UI Elements
const newsEl = document.getElementById("news-container");
const favEl = document.getElementById("favorites-container");
const searchInput = document.getElementById("searchInput");
const categorySel = document.getElementById("category");

// Rate limiting protection
let lastApiCall = 0;
const API_THROTTLE_MS = 1000;
async function throttledFetch(url) {
  const now = Date.now();
  const delta = now - lastApiCall;
  if (delta < API_THROTTLE_MS) {
    await new Promise(r => setTimeout(r, API_THROTTLE_MS - delta));
  }
  lastApiCall = Date.now();
  return fetch(url);
}

// Offline support & caching
function cacheArticles(articles, key = 'cached_news') {
  try {
    localStorage.setItem(key, JSON.stringify({ articles, timestamp: Date.now() }));
  } catch {}
}
function getCachedArticles(key = 'cached_news', maxAge = 3600000) {
  try {
    const item = JSON.parse(localStorage.getItem(key) || '');
    if (item && Date.now() - item.timestamp < maxAge) return item.articles;
  } catch {}
  return null;
}
function showOfflineMessage() {
  const div = document.createElement('div');
  div.className = 'offline-indicator';
  div.textContent = "📡 You're offline. Showing cached news.";
  document.body.prepend(div);
  setTimeout(() => div.remove(), 5000);
}
window.addEventListener('offline', showOfflineMessage);
window.addEventListener('online', () => document.querySelector('.offline-indicator')?.remove());

// API Usage Monitor
class APIMonitor {
  constructor() {
    const stored = JSON.parse(localStorage.getItem('api_usage') || '{}');
    const today = new Date().toDateString();
    if (stored.date !== today) {
      this.usage = { date: today, newsdata: 0, rss: 0 };
    } else {
      this.usage = stored;
    }
    this.updateDisplay();
  }
  save() { localStorage.setItem('api_usage', JSON.stringify(this.usage)); }
  record(api) {
    this.usage[api] = (this.usage[api] || 0) + 1;
    this.save();
    this.updateDisplay();
  }
  canCall(api) {
    return api === 'rss' || (this.usage.newsdata < 200);
  }
  updateDisplay() {
    const el = document.getElementById('api-usage-display');
    if (el) el.textContent = `API Usage: NewsData.io ${this.usage.newsdata}/200, RSS ${this.usage.rss}`;
  }
}
const apiMonitor = new APIMonitor();

// Supabase auth guard
const { data: { user } } = await sb.auth.getUser();
if (!user) window.location.href = "index.html";

// UI Handlers
document.getElementById("logoutBtn").addEventListener("click", async () => {
  await sb.auth.signOut();
  window.location.href = "index.html";
});
document.getElementById("searchBtn").addEventListener("click", () => {
  const q = searchInput.value.trim();
  q ? loadSearch(q) : loadTop(categorySel.value);
});
searchInput.addEventListener('keypress', e => { if (e.key === 'Enter') document.getElementById('searchBtn').click(); });
document.getElementById("darkToggle").addEventListener("click", () => {
  document.documentElement.classList.toggle("dark");
});
categorySel.addEventListener("change", () => loadTop(categorySel.value));

// API Keys & URLs
const NEWSDATA_API_KEY = "pub_06cdef5e37f441338ddceee48e379705";
const NEWSDATA_BASE_URL = "https://newsdata.io/api/1";

// Fetch NewsData.io
async function fetchNewsDataIO(endpoint, params = {}) {
  if (!apiMonitor.canCall('newsdata')) return [];
  const url = new URL(`${NEWSDATA_BASE_URL}/${endpoint}`);
  url.searchParams.append('apikey', NEWSDATA_API_KEY);
  Object.entries(params).forEach(([k,v]) => v && url.searchParams.append(k, v));
  try {
    const res = await throttledFetch(url);
    apiMonitor.record('newsdata');
    if (!res.ok) throw new Error();
    const data = await res.json();
    return data.results || [];
  } catch {
    return [];
  }
}

// Fetch Google News RSS via rss2json
async function fetchGoogleNewsRSS(type, params = {}) {
  let rssUrl = 'https://news.google.com';
  if (type === 'search' && params.q) {
    rssUrl += `/rss/search?q=${encodeURIComponent(params.q)}`;
  } else if (type === 'category') {
    const m = { technology:'TECHNOLOGY', business:'BUSINESS', sports:'SPORTS', health:'HEALTH', entertainment:'ENTERTAINMENT', science:'SCIENCE' };
    rssUrl += `/news/rss/headlines/section/topic/${m[params.category]||'WORLD'}?hl=en-US&gl=US&ceid=US:en`;
  } else {
    rssUrl += '/rss?hl=en-US&gl=US&ceid=US:en';
  }
  try {
    const proxy = `https://api.rss2json.com/v1/api.json?rss_url=${encodeURIComponent(rssUrl)}`;
    const res = await fetch(proxy);
    apiMonitor.record('rss');
    const data = await res.json();
    return data.items.map(i => ({
      title: i.title,
      description: i.description,
      link: i.link,
      pubDate: i.pubDate,
      image_url: (i.content.match(/<img[^>]+src="([^">]+)"/)?.[1]) || null,
      source_id: 'google-news'
    }));
  } catch {
    return [];
  }
}

// Loading Indicators
function showLoading(msg = "Loading news...") {
  newsEl.innerHTML = `<div class="loading"><div class="loading-spinner"></div><p>${msg}</p></div>`;
}
function hideLoading() {
  document.querySelector('.loading')?.remove();
}

// Renderers
function formatDate(d) {
  try {
    return new Date(d).toLocaleDateString('en-IN', { year:'numeric', month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' });
  } catch { return ''; }
}
function card(article) {
  const img = article.image_url || "https://via.placeholder.com/640x360?text=No+Image";
  const title = article.title || "Untitled";
  const desc = (article.description||'').replace(/<[^>]*>/g,'').slice(0,150);
  const link = article.link || "#";
  const src = article.source_id || 'Unknown';
  const date = article.pubDate || new Date().toISOString();
  const div = document.createElement('div');
  div.className = 'card';
  div.innerHTML = `
    <img src="${img}" onerror="this.src='https://via.placeholder.com/640x360?text=No+Image'" />
    <div class="pad">
      <div class="meta"><small>${src} • ${formatDate(date)}</small></div>
      <h4>${title}</h4>
      <p>${desc}</p>
      <div class="actions">
        <button class="preview">👁️ Preview</button>
        <a class="more" href="${link}" target="_blank">Read More</a>
        <button class="save">⭐ Save</button>
      </div>
    </div>`;
  div.querySelector('.save').onclick = () => saveFavorite(title, link, img);
  div.querySelector('.preview').onclick = () => showArticlePreview(article);
  return div;
}
function renderNews(arts) {
  if (!arts.length) {
    newsEl.innerHTML = '<div class="no-results">No articles found.</div>';
    return;
  }
  newsEl.innerHTML = '';
  arts.forEach(a => newsEl.appendChild(card(a)));
}

// Favorites
async function saveFavorite(title, url, image) {
  const { error } = await sb.from('favorites').insert([{ user_id:user.id, title, url, image }]);
  if (!error) loadFavorites();
}
async function loadFavorites() {
  const { data } = await sb.from('favorites').select('*').eq('user_id', user.id).order('created_at', { ascending:false });
  favEl.innerHTML = data.length 
    ? data.map(f => `<div class="card"><img src="${f.image||'https://via.placeholder.com/640x360?text=No+Image'}" /><div class="pad"><h4>${f.title}</h4><div class="actions"><a class="more" href="${f.url}" target="_blank">Read</a><button class="del" data-id="${f.id}">🗑️</button></div></div></div>`).join('')
    : '<div class="no-favorites">No saved articles.</div>';
  favEl.querySelectorAll('.del').forEach(btn =>
    btn.onclick = () => removeFavorite(btn.dataset.id)
  );
}
async function removeFavorite(id) {
  await sb.from('favorites').delete().eq('id', id).eq('user_id', user.id);
  loadFavorites();
}

// Article Preview Modal
function showArticlePreview(article) {
  document.querySelector('.modal-overlay')?.remove();
  const modal = document.createElement('div');
  modal.className = 'modal-overlay';
  modal.innerHTML = `
    <div class="modal-content">
      <button class="close-modal">&times;</button>
      <img src="${article.image_url||'https://via.placeholder.com/600x300?text=No+Image'}" onerror="this.src='https://via.placeholder.com/600x300?text=No+Image'" />
      <div class="modal-body">
        <h2>${article.title}</h2>
        <p class="modal-meta">${article.source_id||'Unknown'} • ${formatDate(article.pubDate)}</p>
        <p class="modal-description">${article.description||'No description.'}</p>
        <div class="modal-actions">
          <a href="${article.link}" target="_blank" class="read-full">Read Full Article</a>
          <button class="save-modal">⭐ Save</button>
        </div>
      </div>
    </div>`;
  document.body.appendChild(modal);
  document.body.style.overflow = 'hidden';
  modal.querySelector('.close-modal').onclick = () => {
    modal.remove();
    document.body.style.overflow = 'auto';
  };
  modal.querySelector('.save-modal').onclick = () => {
    saveFavorite(article.title, article.link, article.image_url);
    modal.remove();
    document.body.style.overflow = 'auto';
  };
}

// Load functions
async function loadTop(category = "") {
  showLoading();
  const key = `cached_news_${category||'all'}`;
  const cached = getCachedArticles(key, 600000);
  if (cached) {
    hideLoading();
    renderNews(cached);
    return;
  }
  let arts = await fetchNewsDataIO('latest', { country:'in', category, language:'en', size:10 });
  if (!arts.length) arts = await fetchGoogleNewsRSS(category?'category':'top', { category });
  if (arts.length) cacheArticles(arts, key);
  hideLoading();
  renderNews(arts);
}
async function loadSearch(q) {
  if (!q) return loadTop();
  showLoading(`Searching for "${q}"...`);
  const key = `cached_search_${q.toLowerCase().replace(/\s+/g,'_')}`;
  const cached = getCachedArticles(key, 300000);
  if (cached) {
    hideLoading();
    renderNews(cached);
    return;
  }
  let arts = await fetchNewsDataIO('news', { q, country:'in', language:'en', size:10 });
  if (!arts.length) arts = await fetchGoogleNewsRSS('search', { q });
  if (arts.length) cacheArticles(arts, key);
  hideLoading();
  renderNews(arts);
}

// Initialize
await loadTop();
await loadFavorites();
