const refInput = document.getElementById("ref");
const folderInput = document.getElementById("folder");
const runBtn = document.getElementById("run");
const statusEl = document.getElementById("status");

const refFacesEl = document.getElementById("refFaces");
const matchesEl = document.getElementById("matches");

function clearUI() {
  refFacesEl.innerHTML = "";
  matchesEl.innerHTML = "";
}

function cardImg(title, imgUrl, meta="") {
  const div = document.createElement("div");
  div.className = "imgCard";
  div.innerHTML = `
    <div class="imgTop">
      <div class="imgTitle">${title}</div>
      ${meta ? `<div class="imgMeta">${meta}</div>` : ""}
    </div>
    <img src="${imgUrl}" loading="lazy" />
  `;
  return div;
}

runBtn.addEventListener("click", async () => {
  clearUI();
  statusEl.textContent = "";

  const ref = refInput.files?.[0];
  const vids = folderInput.files;

  if (!ref) {
    statusEl.textContent = "ویدئوی مرجع را انتخاب کن.";
    return;
  }
  if (!vids || vids.length === 0) {
    statusEl.textContent = "یک پوشه ویدئو انتخاب کن (حداقل 1 فایل mp4).";
    return;
  }

  const fd = new FormData();
  fd.append("reference", ref);

  // فقط mp4 ها
  let count = 0;
  for (const f of vids) {
    if ((f.name || "").toLowerCase().endsWith(".mp4")) {
      fd.append("videos", f, f.name);
      count++;
    }
  }
  if (count === 0) {
    statusEl.textContent = "داخل پوشه هیچ mp4 پیدا نشد.";
    return;
  }

  runBtn.disabled = true;
  statusEl.textContent = "در حال پردازش... (ممکنه چند دقیقه طول بکشه)";

  try {
    const res = await fetch("/api/process", { method: "POST", body: fd });
    const data = await res.json();

    if (!data.ok) {
      statusEl.textContent = "خطا: " + (data.error || "نامشخص");
      runBtn.disabled = false;
      return;
    }

    statusEl.textContent = "تمام شد ✅";

    // چهره‌های مرجع
    for (const p of data.reference_faces) {
      refFacesEl.appendChild(cardImg(p.person_id, p.image_url));
    }

    // match ها
    if (data.matches.length === 0) {
      matchesEl.innerHTML = `<div class="empty">هیچ match معتبری پیدا نشد (threshold فعلی سخت‌گیره یا چهره‌ها کم‌کیفیت‌اند).</div>`;
    } else {
      for (const m of data.matches) {
        const meta = `ویدئو: ${m.video_name} | similarity: ${m.similarity.toFixed(3)}`;
        matchesEl.appendChild(cardImg(m.ref_person, m.pair_image_url, meta));
      }
    }

  } catch (e) {
    statusEl.textContent = "خطا در ارتباط با سرور. کنسول را چک کن.";
    console.error(e);
  } finally {
    runBtn.disabled = false;
  }
});
