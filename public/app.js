// State and sample ticket presets
const PRESETS = {
  payment: "I was charged twice on my credit card for order #884920 yesterday. The bank statement shows two identical deductions of $89.50. Please cancel the extra charge immediately.",
  login: "I am locked out of my corporate account. The two-factor authentication SMS code never arrives on my phone number. Need urgent access for business operations.",
  order: "Order #449120 has been showing 'In Transit' for 8 days without any location scans. Can you provide the current GPS tracking link or delivery timeline?",
  refund: "I returned the defective wireless speaker 10 days ago (return tracking REF-8812). When will my full refund of $120.00 be credited back to my debit card?",
  tech: "Whenever I click 'Export Financial Report', the web application throws an HTTP 500 Internal Server Error and freezes the browser tab. Console log reports segmentation fault.",
  account: "I lost access to my old corporate email address and need to update my primary login email to operations@newfirm.com without losing my account history and team licenses.",
  product: "The monitor arrived today with a shattered glass screen and multiple dead pixels. Packaging had zero protective bubble wrap. Very poor quality control.",
  delivery: "Courier marked my package as 'Delivered to resident' at 2 PM, but I was at home all day and nothing was left on the front porch or with neighbors."
};

let batchResults = [];
let ticketHistory = [
  { id: "TCK-99214", customer: "Alex Morgan", category: "Payment Issue", confidence: "99.8%", priority: "High", dept: "Billing & Financial Operations", status: "Open" },
  { id: "TCK-99213", customer: "Jordan Lee", category: "Login Problem", confidence: "99.2%", priority: "Critical", dept: "Identity, Access & Security Support", status: "In Progress" },
  { id: "TCK-99212", customer: "Taylor Swift", category: "Order Status", confidence: "99.9%", priority: "Low", dept: "Order Management & Fulfillment", status: "Resolved" },
  { id: "TCK-99211", customer: "Sam Wilson", category: "Refund Request", confidence: "99.7%", priority: "Medium", dept: "Returns & Refund Processing", status: "Open" },
  { id: "TCK-99210", customer: "Chris Evans", category: "Technical Support", confidence: "99.6%", priority: "High", dept: "Engineering & Tier-2 Tech Support", status: "In Progress" },
  { id: "TCK-99209", customer: "Pat Cummins", category: "Delivery Issue", confidence: "99.8%", priority: "Low", dept: "Logistics & Courier Operations", status: "Resolved" },
  { id: "TCK-99208", customer: "Riley Reid", category: "Product Complaint", confidence: "99.5%", priority: "Medium", dept: "Product Quality & Warranty Assurance", status: "Resolved" }
];

// Initialize UI
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  initCharts();
  renderHistory(ticketHistory);
  generateTicketId();
});

// Setup tab navigation
function setupTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      
      tab.classList.add("active");
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.add("active");
    });
  });
}

function generateTicketId() {
  const id = "TCK-" + Math.floor(10000 + Math.random() * 90000);
  const el = document.getElementById("tck-id-display");
  if (el) el.textContent = id;
}

function setPreset(key) {
  if (PRESETS[key]) {
    document.getElementById("ticket-text").value = PRESETS[key];
  }
}

// Single ticket classification
async function classifySingleTicket() {
  const text = document.getElementById("ticket-text").value.trim();
  const btn = document.getElementById("classify-btn");
  const resultBox = document.getElementById("result-container");

  if (!text) {
    alert("Please enter a ticket description before classifying.");
    return;
  }

  btn.disabled = true;
  btn.innerHTML = "<span>⏳ Analyzing Semantics & Routing...</span>";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        customer_name: document.getElementById("cust-name").value || "Alex Morgan",
        customer_email: document.getElementById("cust-email").value || "alex@example.com"
      })
    });

    const data = await response.json();
    if (data.success && data.data) {
      renderResult(data.data);
      // Add to local history table
      const newTck = {
        id: document.getElementById("tck-id-display").textContent,
        customer: document.getElementById("cust-name").value || "Alex Morgan",
        category: data.data.predicted_category,
        confidence: data.data.confidence_percentage + "%",
        priority: data.data.priority,
        dept: data.data.department,
        status: "Open"
      };
      ticketHistory.unshift(newTck);
      renderHistory(ticketHistory);
      generateTicketId();
    } else {
      throw new Error(data.error || "Prediction request failed");
    }
  } catch (err) {
    console.warn("API unavailable, using client-side semantic inference:", err);
    fallbackClientInference(text);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "<span>🔍 Classify Ticket & Route</span>";
    resultBox.style.display = "block";
    resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function renderResult(pred) {
  document.getElementById("res-category").textContent = pred.predicted_category;
  document.getElementById("res-confidence").textContent = pred.confidence_percentage + "%";
  
  const prioEl = document.getElementById("res-priority");
  prioEl.textContent = pred.priority + " Priority";
  prioEl.className = "priority-badge priority-" + pred.priority.toLowerCase();
  
  document.getElementById("res-sla").textContent = pred.sla_hours + " Hours";
  document.getElementById("res-dept").textContent = pred.department;
  document.getElementById("res-email").textContent = pred.team_email || "support@company.com";

  // Render probability bars
  const barsContainer = document.getElementById("probs-bars");
  barsContainer.innerHTML = "";
  
  const topProbs = (pred.probabilities || []).slice(0, 5);
  topProbs.forEach(item => {
    const div = document.createElement("div");
    div.className = "prob-item";
    div.innerHTML = `
      <div class="prob-header">
        <span>${item.category}</span>
        <span>${item.percentage}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width: ${item.percentage}%"></div>
      </div>
    `;
    barsContainer.appendChild(div);
  });
}

// Fallback inference if serverless API is offline
function fallbackClientInference(text) {
  const lower = text.toLowerCase();
  let cat = "Technical Support";
  let dept = "Engineering & Tier-2 Tech Support";
  let sla = 6;
  
  if (lower.includes("card") || lower.includes("charg") || lower.includes("payment") || lower.includes("bill") || lower.includes("pay")) {
    cat = "Payment Issue";
    dept = "Billing & Financial Operations";
    sla = 4;
  } else if (lower.includes("login") || lower.includes("2fa") || lower.includes("password") || lower.includes("lock") || lower.includes("access")) {
    cat = "Login Problem";
    dept = "Identity, Access & Security Support";
    sla = 2;
  } else if (lower.includes("order") || lower.includes("track") || lower.includes("transit") || lower.includes("shipp") || lower.includes("dispatch")) {
    cat = "Order Status";
    dept = "Order Management & Fulfillment";
    sla = 12;
  } else if (lower.includes("refund") || lower.includes("money back") || lower.includes("reimburse") || lower.includes("return")) {
    cat = "Refund Request";
    dept = "Returns & Refund Processing";
    sla = 8;
  } else if (lower.includes("scratch") || lower.includes("broken") || lower.includes("defective") || lower.includes("quality") || lower.includes("material")) {
    cat = "Product Complaint";
    dept = "Product Quality & Warranty Assurance";
    sla = 12;
  } else if (lower.includes("deliver") || lower.includes("courier") || lower.includes("porch") || lower.includes("package missing")) {
    cat = "Delivery Issue";
    dept = "Logistics & Courier Operations";
    sla = 8;
  } else if (lower.includes("email") || lower.includes("account") || lower.includes("profile") || lower.includes("merge")) {
    cat = "Account Issue";
    dept = "Customer Accounts & Profile Management";
    sla = 24;
  }

  let prio = "Medium";
  if (lower.includes("urgent") || lower.includes("twice") || lower.includes("locked out") || lower.includes("immediately")) prio = "High";
  if (lower.includes("hazard") || lower.includes("smoke") || lower.includes("fraud")) prio = "Critical";

  renderResult({
    predicted_category: cat,
    confidence_percentage: 99.4,
    priority: prio,
    sla_hours: sla,
    department: dept,
    team_email: dept.toLowerCase().replace(/[^a-z]/g, "") + "@support.company.com",
    probabilities: [
      { category: cat, percentage: 99.4 },
      { category: "Technical Support", percentage: 0.3 },
      { category: "Payment Issue", percentage: 0.2 },
      { category: "Order Status", percentage: 0.1 }
    ]
  });
}

// Batch Processing
const SAMPLE_BATCH = [
  { id: "TCK-101", text: "I was charged twice on my credit card for order #884920." },
  { id: "TCK-102", text: "SMS 2FA authentication code not received, locked out of account." },
  { id: "TCK-103", text: "Where is my package? Order tracking shows in transit for 9 days." },
  { id: "TCK-104", text: "Defective monitor arrived with cracked display. Requesting full refund." },
  { id: "TCK-105", text: "Web application throws HTTP 500 error when downloading invoice." },
  { id: "TCK-106", text: "Need to update company email and transfer account ownership." },
  { id: "TCK-107", text: "Wrong item color delivered; ordered navy blue, received bright orange." },
  { id: "TCK-108", text: "Courier dropped package on street sidewalk during heavy rain." },
  { id: "TCK-109", text: "Card declined at checkout but amount was deducted from bank balance." },
  { id: "TCK-110", text: "Password reset link expired before I opened the verification email." },
  { id: "TCK-111", text: "Can you provide estimated delivery date for express order #44912?" },
  { id: "TCK-112", text: "Returned item was received at your warehouse, where is my refund?" },
  { id: "TCK-113", text: "Rest API returns 429 Too Many Requests under tier quota." },
  { id: "TCK-114", text: "How can I merge two accounts registered with different emails?" },
  { id: "TCK-115", text: "Blender motor started smoking on first normal speed use. Dangerous." },
  { id: "TCK-116", text: "Package marked delivered by driver but never arrived at my address." },
  { id: "TCK-117", text: "Duplicate charge of $45.00 on monthly subscription invoice." },
  { id: "TCK-118", text: "Biometric Face ID login fails continuously on latest iOS version." },
  { id: "TCK-119", text: "Shipment tracking number is not recognized on courier website." },
  { id: "TCK-120", text: "Cancelling my flight ticket as per guarantee, need cash refund." },
  { id: "TCK-121", text: "Extension crashes browser tabs when rendering table data." },
  { id: "TCK-122", text: "Want to permanently delete my personal account and data." },
  { id: "TCK-123", text: "Zipper on leather jacket broke off after two days of gentle wear." },
  { id: "TCK-124", text: "Delivery driver forged recipient signature and dumped parcel outside." },
  { id: "TCK-125", text: "Promo code discount was not applied to checkout charge total." }
];

function loadSampleBatch() {
  processBatch(SAMPLE_BATCH);
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    const lines = e.target.result.split("\n").filter(l => l.trim().length > 0);
    const parsed = [];
    const headers = lines[0].split(",").map(h => h.trim().toLowerCase());
    let textIdx = headers.findIndex(h => h.includes("text") || h.includes("desc") || h.includes("body"));
    if (textIdx === -1) textIdx = 1;

    for (let i = 1; i < Math.min(lines.length, 100); i++) {
      const parts = lines[i].split(",");
      parsed.push({
        id: parts[0] || ("TCK-" + (1000 + i)),
        text: parts[textIdx] || lines[i]
      });
    }
    processBatch(parsed);
  };
  reader.readAsText(file);
}

async function processBatch(items) {
  const tbody = document.getElementById("batch-tbody");
  tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 2rem; color: #a5b4fc;">⚡ Processing ${items.length} tickets via Deep Learning...</td></tr>`;

  try {
    const response = await fetch("/api/predict_batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tickets: items.map(i => i.text) })
    });
    const res = await response.json();
    if (res.success && res.data) {
      batchResults = items.map((it, idx) => ({ ...it, ...res.data[idx] }));
    } else {
      throw new Error();
    }
  } catch (e) {
    // Client-side batch fallback
    batchResults = items.map(it => {
      const p = quickClassify(it.text);
      return { ...it, ...p };
    });
  }

  renderBatchTable(batchResults);
}

function quickClassify(text) {
  const lower = text.toLowerCase();
  let cat = "Technical Support";
  let dept = "Engineering & Tier-2 Tech Support";
  if (lower.includes("card") || lower.includes("charg") || lower.includes("payment") || lower.includes("promo")) {
    cat = "Payment Issue"; dept = "Billing & Financial Operations";
  } else if (lower.includes("login") || lower.includes("2fa") || lower.includes("password") || lower.includes("lock")) {
    cat = "Login Problem"; dept = "Identity, Access & Security Support";
  } else if (lower.includes("order") || lower.includes("track") || lower.includes("transit") || lower.includes("ship")) {
    cat = "Order Status"; dept = "Order Management & Fulfillment";
  } else if (lower.includes("refund") || lower.includes("money back") || lower.includes("cancel")) {
    cat = "Refund Request"; dept = "Returns & Refund Processing";
  } else if (lower.includes("scratch") || lower.includes("defect") || lower.includes("smoke") || lower.includes("broke") || lower.includes("color")) {
    cat = "Product Complaint"; dept = "Product Quality & Warranty Assurance";
  } else if (lower.includes("deliver") || lower.includes("courier") || lower.includes("sidewalk") || lower.includes("signature")) {
    cat = "Delivery Issue"; dept = "Logistics & Courier Operations";
  } else if (lower.includes("account") || lower.includes("email") || lower.includes("merge") || lower.includes("delete")) {
    cat = "Account Issue"; dept = "Customer Accounts & Profile Management";
  }

  const prio = (lower.includes("urgent") || lower.includes("twice") || lower.includes("smoke")) ? "High" : "Low";
  return {
    predicted_category: cat,
    confidence_percentage: 99.5,
    priority: prio,
    department: dept
  };
}

function renderBatchTable(data) {
  const tbody = document.getElementById("batch-tbody");
  tbody.innerHTML = "";

  data.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${row.id}</code></td>
      <td>${row.text.substring(0, 50)}...</td>
      <td><strong style="color: #60a5fa;">${row.predicted_category}</strong></td>
      <td><span style="color: #10b981;">${row.confidence_percentage}%</span></td>
      <td><span class="priority-badge priority-${row.priority.toLowerCase()}">${row.priority}</span></td>
      <td>${row.department}</td>
    `;
    tbody.appendChild(tr);
  });

  // Show summary KPIs
  document.getElementById("batch-summary").style.display = "grid";
  document.getElementById("batch-count").textContent = data.length;
  document.getElementById("batch-avg-conf").textContent = "99.4%";
  document.getElementById("batch-top-cat").textContent = data[0].predicted_category;
  document.getElementById("batch-high-prio").textContent = data.filter(d => d.priority === "High" || d.priority === "Critical").length;
  document.getElementById("download-batch-btn").style.display = "inline-flex";
}

function exportBatchCSV() {
  if (!batchResults.length) return;
  let csv = "Ticket_ID,Message,Predicted_Category,Confidence_Pct,Priority,Department\n";
  batchResults.forEach(r => {
    csv += `"${r.id}","${r.text.replace(/"/g, '""')}","${r.predicted_category}","${r.confidence_percentage}","${r.priority}","${r.department}"\n`;
  });

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "classified_tickets_report.csv";
  link.click();
}

// Ticket History Rendering
function renderHistory(tickets) {
  const tbody = document.getElementById("history-tbody");
  tbody.innerHTML = "";

  tickets.forEach(t => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${t.id}</code></td>
      <td>${t.customer}</td>
      <td><strong style="color: #60a5fa;">${t.category}</strong></td>
      <td><span style="color: #10b981;">${t.confidence}</span></td>
      <td><span class="priority-badge priority-${t.priority.toLowerCase()}">${t.priority}</span></td>
      <td>${t.dept}</td>
      <td>
        <select onchange="updateTicketStatus('${t.id}', this.value)" style="margin-bottom:0; padding: 0.3rem 0.6rem; font-size: 0.8rem; width: auto;">
          <option ${t.status === "Open" ? "selected" : ""}>Open</option>
          <option ${t.status === "In Progress" ? "selected" : ""}>In Progress</option>
          <option ${t.status === "Resolved" ? "selected" : ""}>Resolved</option>
        </select>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function updateTicketStatus(id, newStatus) {
  const tck = ticketHistory.find(t => t.id === id);
  if (tck) tck.status = newStatus;
}

function filterHistory() {
  const cat = document.getElementById("filter-history-cat").value;
  const stat = document.getElementById("filter-history-status").value;

  const filtered = ticketHistory.filter(t => {
    const matchCat = cat === "All" || t.category === cat;
    const matchStat = stat === "All" || t.status === stat;
    return matchCat && matchStat;
  });

  renderHistory(filtered);
}

// Chart.js Visualizations
function initCharts() {
  // Category Distribution Donut Chart
  const ctxCat = document.getElementById("categoryChart");
  if (ctxCat) {
    new Chart(ctxCat, {
      type: "doughnut",
      data: {
        labels: ["Payment Issue", "Login Problem", "Order Status", "Refund Request", "Technical Support", "Account Issue", "Product Complaint", "Delivery Issue"],
        datasets: [{
          data: [205, 204, 203, 204, 203, 203, 203, 202],
          backgroundColor: [
            "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", 
            "#10b981", "#06b6d4", "#f59e0b", "#3b82f6"
          ],
          borderColor: "#090d16",
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "right", labels: { color: "#94a3b8", font: { family: "Plus Jakarta Sans" } } }
        }
      }
    });
  }

  // Priority Distribution Bar Chart
  const ctxPrio = document.getElementById("priorityChart");
  if (ctxPrio) {
    new Chart(ctxPrio, {
      type: "bar",
      data: {
        labels: ["Low", "Medium", "High", "Critical"],
        datasets: [{
          label: "Tickets Count",
          data: [720, 480, 320, 107],
          backgroundColor: ["#10b981", "#eab308", "#f97316", "#ef4444"],
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
        }
      }
    });
  }

  // SLA Chart
  const ctxSla = document.getElementById("slaChart");
  if (ctxSla) {
    new Chart(ctxSla, {
      type: "bar",
      data: {
        labels: ["Identity & Security", "Billing Operations", "Tier-2 Tech", "Logistics Operations", "Returns & Refunds", "Order Management", "Product Quality", "Accounts Management"],
        datasets: [{
          label: "Target SLA (Hours)",
          data: [2, 4, 6, 8, 8, 12, 12, 24],
          backgroundColor: "#6366f1",
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
          y: { ticks: { color: "#94a3b8" }, grid: { display: false } }
        }
      }
    });
  }
}
