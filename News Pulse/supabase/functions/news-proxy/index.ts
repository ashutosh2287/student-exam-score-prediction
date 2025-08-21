// supabase/functions/news-proxy/index.ts
// Deno Edge Function
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";

const cors = {
  "Access-Control-Allow-Origin": "*", // prod me apna domain daalna
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Content-Type": "application/json",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: cors });
  }

  const NEWSAPI_KEY = Deno.env.get("NEWSAPI_KEY");
  if (!NEWSAPI_KEY) {
    return new Response(JSON.stringify({ error: "Missing NEWSAPI_KEY" }), { status: 500, headers: cors });
  }

  const url = new URL(req.url);
  const type = url.searchParams.get("type") ?? "top"; // "top" | "search"
  const country = url.searchParams.get("country") ?? "in";
  const category = url.searchParams.get("category") ?? "";
  const q = url.searchParams.get("q") ?? "";

  let endpoint = "";
  if (type === "search") {
    endpoint = `https://newsapi.org/v2/everything?q=${encodeURIComponent(q)}&pageSize=30&sortBy=publishedAt&language=en`;
  } else {
    endpoint = `https://newsapi.org/v2/top-headlines?country=${country}${category ? `&category=${category}` : ""}&pageSize=30`;
  }

  const upstream = await fetch(endpoint, { headers: { "X-Api-Key": NEWSAPI_KEY } });
  const data = await upstream.json();

  return new Response(JSON.stringify(data), { headers: cors, status: upstream.status });
});
