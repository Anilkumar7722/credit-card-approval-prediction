// ============================================================
// script.js — shared site behaviour (nav, theme, reveal, flash)
// ============================================================

(function () {
  "use strict";

  /* ---------- Theme toggle (persisted for the session) ---------- */
  const root = document.documentElement;
  const themeToggle = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    if (themeIcon) {
      themeIcon.className = theme === "dark" ? "fa-solid fa-moon" : "fa-solid fa-sun";
    }
  }

  const savedTheme = sessionStorage.getItem("ci-theme") || "dark";
  applyTheme(savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", function () {
      const current = root.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      applyTheme(next);
      sessionStorage.setItem("ci-theme", next);
    });
  }

  /* ---------- Mobile nav ---------- */
  const navToggle = document.getElementById("navToggle");
  const navLinks = document.getElementById("navLinks");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
      const isOpen = navLinks.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
    });
    navLinks.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => navLinks.classList.remove("open"));
    });
  }

  /* ---------- Sticky nav shadow on scroll ---------- */
  const nav = document.getElementById("ciNav");
  if (nav) {
    window.addEventListener("scroll", function () {
      nav.style.boxShadow = window.scrollY > 12 ? "0 8px 24px -12px rgba(0,0,0,0.4)" : "none";
    });
  }

  /* ---------- Flash dismiss ---------- */
  document.querySelectorAll(".ci-flash-close").forEach((btn) => {
    btn.addEventListener("click", function () {
      const flash = btn.closest(".ci-flash");
      if (flash) {
        flash.style.opacity = "0";
        setTimeout(() => flash.remove(), 250);
      }
    });
  });
  document.querySelectorAll(".ci-flash").forEach((flash) => {
    setTimeout(() => {
      flash.style.transition = "opacity .4s ease";
      flash.style.opacity = "0";
      setTimeout(() => flash.remove(), 400);
    }, 6000);
  });

  /* ---------- Scroll reveal ---------- */
  const revealEls = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && revealEls.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in-view");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15 }
    );
    revealEls.forEach((el) => observer.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("in-view"));
  }

  /* ---------- FAQ accordion ---------- */
  document.querySelectorAll(".faq-item").forEach((item) => {
    const question = item.querySelector(".faq-q");
    if (!question) return;
    question.addEventListener("click", () => {
      const wasOpen = item.classList.contains("open");
      item.closest(".faq-list")?.querySelectorAll(".faq-item").forEach((i) => i.classList.remove("open"));
      if (!wasOpen) item.classList.add("open");
    });
  });

  /* ---------- Animated counters (perf strip / stats) ---------- */
  function animateCount(el) {
    const target = parseFloat(el.dataset.count);
    const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals, 10) : 0;
    const duration = 1400;
    const start = performance.now();

    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = target * eased;
      el.textContent = value.toFixed(decimals);
      if (progress < 1) requestAnimationFrame(tick);
      else el.textContent = target.toFixed(decimals);
    }
    requestAnimationFrame(tick);
  }

  const counters = document.querySelectorAll("[data-count]");
  if ("IntersectionObserver" in window && counters.length) {
    const counterObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animateCount(entry.target);
            counterObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    counters.forEach((el) => counterObserver.observe(el));
  } else {
    counters.forEach(animateCount);
  }

  /* ---------- Perf bar fill widths (triggered on view) ---------- */
  const perfBars = document.querySelectorAll(".perf-bar-fill");
  if ("IntersectionObserver" in window && perfBars.length) {
    const barObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.style.width = entry.target.dataset.width || "0%";
            barObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.3 }
    );
    perfBars.forEach((el) => barObserver.observe(el));
  }
})();
