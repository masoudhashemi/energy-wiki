// Text, critiques and chart points are generated from the saved run by
// build_exp5_walkthrough.py. This script only switches the visible revision.
(() => {
  const buttons = Array.from(document.querySelectorAll('[data-exp5-step]'));
  const panels = Array.from(document.querySelectorAll('.exp5-panel'));
  const status = document.getElementById('exp5-selection');
  function selectRevision(step) {
    if (!buttons.some(button => button.dataset.exp5Step === step)) return;
    buttons.forEach(button => {
      const selected = button.dataset.exp5Step === step;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    panels.forEach(panel => { panel.hidden = panel.id !== 'exp5-panel-' + step; });
    status.textContent = step === '0' ? 'Showing the original seed.' :
      `Showing revision ${step}${step === '10' ? ' (final document)' : ''}.`;
  }
  buttons.forEach(button => button.addEventListener('click', () => selectRevision(button.dataset.exp5Step)));
})();
