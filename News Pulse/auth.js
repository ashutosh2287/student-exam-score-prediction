// auth.js
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = "https://buganeskvrtcnzdumjzo.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ1Z2FuZXNrdnJ0Y256ZHVtanpvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU3NTU1MjQsImV4cCI6MjA3MTMzMTUyNH0.vFRUBLKs71S7TtnLm6jNsRNVI-FQtA2SeoJNdkjZsOk";
const sb = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Already logged in? go dashboard
const { data: sess } = await sb.auth.getSession();
if (sess?.session) window.location.href = "dashboard.html";

document.getElementById("signupBtn").addEventListener("click", async () => {
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;
  const { error } = await sb.auth.signUp({ email, password });
  alert(error ? error.message : "Signup successful! Check your email.");
});

document.getElementById("loginBtn").addEventListener("click", async () => {
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;
  const { error } = await sb.auth.signInWithPassword({ email, password });
  if (error) return alert(error.message);
  window.location.href = "dashboard.html";
});
