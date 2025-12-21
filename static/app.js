const refInput = document.getElementById("ref");
const folderInput = document.getElementById("folder");
const runBtn = document.getElementById("run");
const clearBtn = document.getElementById("clear");
const statusEl = document.getElementById("status");
const sideStatusEl = document.getElementById("sideStatus");

const refFacesEl = document.getElementById("refFaces");
const matchesEl = document.getElementById("matches");

function setStatus(text, type = "neutral") {
  statusEl.textContent = text || "";
  statusEl.classList.remove("status--ok", "status--err");

  if (type === "ok") statusEl.classList.add("status--ok");
  if (type === "err") statusEl.classList.add("status--err");

  if (sideStatusEl) sideStatusEl.textContent = text ? text.replace("","").trim() : "";
}

function clearUI() {
  refFacesEl.innerHTML = "";
  matchesEl.innerHTML = "";
}

function cardImg(title, imgUrl, meta = "") {
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

clearBtn?.addEventListener("click", () => {
  clearUI();
});

runBtn.addEventListener("click", async () => {
  clearUI();
  setStatus("");

  const ref = refInput.files?.[0];
  const vids = folderInput.files;
  const fd = new FormData();
  fd.append("reference", ref);

  let count = 0;
  for (const f of vids) {
    if ((f.name || "").toLowerCase().endsWith(".mp4")) {
      fd.append("videos", f, f.name);
      count++;
    }
  }

  runBtn.disabled = true;

  try {
    const res = await fetch("/api/process", { method: "POST", body: fd });
    const data = await res.json();

    if (!data.ok) {
      runBtn.disabled = false;
      return;
    }


    for (const p of data.reference_faces) {
      refFacesEl.appendChild(cardImg(p.person_id, p.image_url));
    }

    if (data.matches.length === 0) {
      matchesEl.innerHTML =
        `<div class="empty"> not match.</div>`;
    } else {
      for (const m of data.matches) {
        const meta = `: ${m.video_name} | similarity: ${m.similarity.toFixed(3)}`;
        matchesEl.appendChild(cardImg(m.ref_person, m.pair_image_url, meta));
      }
    }

  } catch (e) {
    console.error(e);
  } finally {
    runBtn.disabled = false;
  }
});
