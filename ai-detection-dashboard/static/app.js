
(() => {
  const data = {
    detectionTypes: [
      { id: "deepfake", name: "Deepfake Detection", description: "Detects manipulated content.", supported_formats: ["MP4", "JPG"], icon: "fas fa-video" },
      { id: "object", name: "Object Detection", description: "Identifies objects in media.", supported_formats: ["JPG", "PNG"], icon: "fas fa-eye" },
      { id: "fraud", name: "Fraud Detection", description: "Flags financial fraud.", supported_formats: ["CSV", "PDF"], icon: "fas fa-exclamation-triangle" }
    ],
    statistics: { total_detections: 0, deepfake_detections: 0, object_detections: 0, fraud_detections: 0, reports_generated: 0 },
    sampleDetections: [],
    evidenceReports: [],
    chartData: {
      activity_timeline: { labels: [], detections: [], reports: [] },
      detection_distribution: { labels: ["Deepfake", "Object", "Fraud"], data: [0, 0, 0] }
    }
  };

  let selectedFile = null;

  function formatConfidence(c) { return `${(c * 100).toFixed(1)}%`; }
  function formatDate(iso) { return new Date(iso).toLocaleString(); }
  function randomId() { return Math.random().toString(36).slice(2, 10); }

  function renderStats() {
    const grid = document.getElementById("statsGrid");
    if (!grid) return;
    const stats = [
      { label: "Total Detections", value: data.statistics.total_detections, icon: "fas fa-search" },
      { label: "Deepfake", value: data.statistics.deepfake_detections, icon: "fas fa-video" },
      { label: "Object", value: data.statistics.object_detections, icon: "fas fa-eye" },
      { label: "Fraud", value: data.statistics.fraud_detections, icon: "fas fa-exclamation-triangle" }
    ];
    grid.innerHTML = stats.map(s => `
      <div class="stat-card">
        <div><i class="${s.icon}"></i> <span>${s.value}</span></div>
        <div>${s.label}</div>
      </div>`).join('');
  }

  function renderDetectionsTable() {
    const tbody = document.querySelector("#detectionsTable tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    data.sampleDetections.forEach(d => {
      const reportExists = data.evidenceReports.some(r => r.detection_id === d.id);
      const row = tbody.insertRow();
      row.innerHTML = `
        <td>${d.id}</td><td>${d.type}</td><td>${d.filename}</td>
        <td>${formatConfidence(d.confidence)}</td><td>${formatDate(d.timestamp)}</td>
        <td><button class="action-btn" data-id="${d.id}" data-action="${reportExists ? "view" : "generate"}">${reportExists ? "View" : "Generate"}</button></td>`;
    });
  }

  function renderReportsTable() {
    const tbody = document.querySelector("#reportsTable tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    data.evidenceReports.forEach(r => {
      const row = tbody.insertRow();
      row.innerHTML = `
        <td>${r.id}</td><td>${r.type}</td><td>${formatDate(r.generated_at)}</td>
        <td>${r.status}</td><td><button class="download-btn" data-url="${r.pdfBlobUrl}">Download</button></td>`;
    });
  }

  async function performActualAnalysis(type, file) {
    const progress = document.getElementById("progressBar");
    const result = document.getElementById("analysisResult");
    progress.classList.remove("hidden");
    progress.value = 0;
    result.innerHTML = "";

    const formData = new FormData();
    formData.append('file', file);
    formData.append('detection_type', type);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        if (progress.value < 90) {
          progress.value += 10;
        }
      }, 200);

      const response = await fetch('/detection', {
        method: 'POST',
        body: formData
      });

      clearInterval(progressInterval);
      progress.value = 100;

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result_data = await response.json();
      
      if (result_data.success) {
        const detection = {
          id: result_data.detection_id || Date.now(),
          type: result_data.detection_type || type,
          filename: result_data.filename || file.name,
          confidence: result_data.confidence || 0.85,
          timestamp: result_data.timestamp || new Date().toISOString()
        };

        data.sampleDetections.unshift(detection);
        data.statistics.total_detections++;
        data.statistics[`${type}_detections`]++;

        renderStats();
        renderDetectionsTable();
        showAnalysisResult(detection);
      } else {
        result.innerHTML = `<div class="error">Error: ${result_data.error || 'Unknown error occurred'}</div>`;
      }
    } catch (error) {
      console.error('Analysis error:', error);
      result.innerHTML = `<div class="error">Analysis failed: ${error.message}</div>`;
    } finally {
      setTimeout(() => progress.classList.add("hidden"), 1000);
    }
  }

  function showAnalysisResult(det) {
    const el = document.getElementById("analysisResult");
    if (!el) return;
    el.innerHTML = `
      <div><strong>ID:</strong> ${det.id}</div>
      <div><strong>Type:</strong> ${det.type}</div>
      <div><strong>Confidence:</strong> ${formatConfidence(det.confidence)}</div>
      <button id="genReport">Generate Report</button>`;
    document.getElementById("genReport").onclick = () => generateReport(det);
  }

  async function generateReport(detection) {
    try {
      const response = await fetch('/reports/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          detection_id: detection.id
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        // Create a local representation for the UI
        const reportId = result.report_id || `EVD-${randomId().toUpperCase()}`;
        
        // Try to download the PDF from backend
        const pdfUrl = result.pdf_url || `/reports/download/${result.report_id}`;
        
        data.evidenceReports.unshift({
          id: reportId,
          detection_id: detection.id,
          generated_at: result.timestamp || new Date().toISOString(),
          type: detection.type,
          status: "verified",
          pdfBlobUrl: pdfUrl
        });
        
        data.statistics.reports_generated++;
        renderReportsTable();
        renderDetectionsTable();
        
        alert('Report generated successfully!');
        
        // Open the report
        window.open(pdfUrl, '_blank');
      } else {
        throw new Error(result.error || 'Failed to generate report');
      }
    } catch (error) {
      console.error('Report generation error:', error);
      alert('Failed to generate report: ' + error.message);
      
      // Fallback to client-side PDF generation
      generateClientSidePDF(detection);
    }
  }

  function generateClientSidePDF(detection) {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const reportId = `EVD-${randomId().toUpperCase()}`;

    doc.text("AI Detection Report", 10, 10);
    doc.text(`ID: ${detection.id}`, 10, 20);
    doc.text(`Type: ${detection.type}`, 10, 30);
    doc.text(`Confidence: ${formatConfidence(detection.confidence)}`, 10, 40);
    doc.text(`Timestamp: ${formatDate(detection.timestamp)}`, 10, 50);

    const blob = doc.output("blob");
    const url = URL.createObjectURL(blob);

    data.evidenceReports.unshift({
      id: reportId,
      detection_id: detection.id,
      generated_at: new Date().toISOString(),
      type: detection.type,
      status: "verified",
      pdfBlobUrl: url
    });
    
    data.statistics.reports_generated++;
    renderReportsTable();
    renderDetectionsTable();
    window.open(url);
  }

  function initUI() {
    const select = document.getElementById("detectionType");
    const input = document.getElementById("fileInput");
    const btn = document.getElementById("analyzeBtn");
    if (!select || !input || !btn) return;

    data.detectionTypes.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.id;
      opt.textContent = d.name;
      select.appendChild(opt);
    });

    input.addEventListener("change", e => {
      selectedFile = e.target.files[0];
      btn.disabled = !(selectedFile && select.value);
    });
    
    select.addEventListener("change", () => {
      btn.disabled = !(selectedFile && select.value);
    });
    
    btn.addEventListener("click", () => {
      performActualAnalysis(select.value, selectedFile);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderStats();
    renderDetectionsTable();
    renderReportsTable();
    initUI();

    // Secret hotkey: Shift+Alt+Enter to show author name
    document.addEventListener("keydown", function(e) {
      if (e.shiftKey && e.altKey && (e.key === "Enter" || e.code === "Enter")) {
        alert("Author: Akshat Jasrotia\nGithub: akshatjasrotia85");
      }
    });

    document.body.addEventListener("click", e => {
      if (e.target.classList.contains("action-btn")) {
        const id = +e.target.dataset.id;
        const det = data.sampleDetections.find(d => d.id === id);
        if (!det) return;
        if (e.target.dataset.action === "generate") {
          generateReport(det);
        } else {
          const rep = data.evidenceReports.find(r => r.detection_id === det.id);
          if (rep) window.open(rep.pdfBlobUrl);
        }
      }

      if (e.target.classList.contains("download-btn")) {
        const url = e.target.dataset.url;
        if (url) window.open(url);
      }
    });
  });
})();