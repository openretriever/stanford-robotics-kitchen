const eventSources = {
  dispatch: {label: 'Dispatch', color: '#326cb0'},
  motion: {label: 'Motion', color: '#8562ac'},
  simulation: {label: 'Physics', color: '#a96b1f'},
  memory: {label: 'Memory', color: '#18837d'},
  verification: {label: 'Verifier', color: '#517c38'},
};
let timelineKey = '', timelineEvents = [], lastEventId = 0;
let revealMemory = () => {};

function setupConsole() {
  const robotCamera = document.createElement('button');
  robotCamera.dataset.camera = 'Aloha';
  robotCamera.title = robotCamera.ariaLabel = 'Show Mobile ALOHA';
  robotCamera.setAttribute('aria-pressed', 'false');
  robotCamera.innerHTML = '<i data-lucide="bot"></i>';
  document.getElementById('presentation').before(robotCamera);
  const drawerTab = document.createElement('button');
  drawerTab.id = 'tab-drawers';
  drawerTab.dataset.tab = 'drawers';
  drawerTab.setAttribute('role', 'tab');
  drawerTab.setAttribute('aria-controls', 'drawers');
  drawerTab.textContent = 'Drawers';
  document.querySelector('.tabs').append(drawerTab);
  const drawerPanel = document.createElement('section');
  drawerPanel.id = 'drawers';
  drawerPanel.className = 'tab-content drawer-inspector';
  drawerPanel.setAttribute('role', 'tabpanel');
  drawerPanel.setAttribute('aria-labelledby', 'tab-drawers');
  drawerPanel.hidden = true;
  drawerPanel.innerHTML = '<div class="drawer-toolbar"><select id="drawer-choice" aria-label="Drawer"></select><button class="icon" id="close-drawers" title="Close all drawers" aria-label="Close all drawers"><i data-lucide="minimize-2"></i></button></div><div class="drawer-opening"><label for="drawer-opening">Opening</label><output id="drawer-percent">0%</output><input id="drawer-opening" type="range" min="0" max="100" step="1" value="0"></div><div id="drawer-list"></div>';
  document.getElementById('plan').after(drawerPanel);
  document.getElementById('close-drawers').onclick = () => command('drawers-close');
  document.getElementById('drawer-choice').onchange = () => renderDrawers(state);
  document.getElementById('drawer-opening').oninput = e => document.getElementById('drawer-percent').textContent = e.target.value + '%';
  document.getElementById('drawer-opening').onchange = e => command('drawer', {id:document.getElementById('drawer-choice').value,fraction:Number(e.target.value)/100});
  const tour = document.createElement('button');
  tour.id = 'camera-tour';
  tour.title = tour.ariaLabel = 'Start full kitchen tour';
  tour.innerHTML = '<i data-lucide="film"></i>';
  tour.onclick = () => command('tour');
  document.querySelector('.camera').append(tour);
  const speedBadge = document.createElement('div');
  speedBadge.className = 'playback-badge';
  speedBadge.innerHTML = '<small>SIMULATION SPEED</small><strong id="playback-speed">4x</strong>';
  document.querySelector('.viewport').append(speedBadge);
  const memory = document.getElementById('memory');
  const anchor = document.createComment('memory dock');
  memory.before(anchor);
  const title = memory.querySelector('.heading-row');
  title.classList.add('memory-header');
  const grip = document.createElement('button');
  grip.className = 'icon memory-grip';
  grip.type = 'button';
  grip.title = grip.ariaLabel = 'Move memory panel';
  grip.innerHTML = '<i data-lucide="grip-vertical"></i>';
  const dock = document.createElement('button');
  dock.className = 'icon';
  dock.type = 'button';
  dock.id = 'memory-dock';
  title.prepend(grip);
  title.append(dock);
  const tools = document.createElement('div');
  tools.className = 'memory-tools';
  tools.append(document.getElementById('search-mode'), document.getElementById('forget-memory'));
  title.after(tools);
  function updateDock() {
    const floating = memory.classList.contains('detached');
    dock.title = dock.ariaLabel = floating ? 'Dock memory panel' : 'Undock memory panel';
    dock.setAttribute('aria-pressed', String(floating));
    dock.innerHTML = `<i data-lucide="${floating ? 'panel-right-close' : 'panels-top-left'}"></i>`;
    lucide.createIcons();
  }
  function place(x, y) {
    memory.style.left = Math.max(8, Math.min(x, document.documentElement.clientWidth - memory.offsetWidth - 8)) + 'px';
    memory.style.top = Math.max(8, Math.min(y, innerHeight - memory.offsetHeight - 8)) + 'px';
  }
  function undock() {
    if (memory.classList.contains('detached')) return;
    const scene = document.querySelector('.viewport').getBoundingClientRect();
    document.body.append(memory);
    memory.classList.add('detached');
    place(scene.right - memory.offsetWidth - 18, scene.top + 64);
    updateDock();
  }
  function dockMemory() {
      memory.classList.remove('detached');
      memory.style.left = memory.style.top = '';
      anchor.after(memory);
      updateDock();
  }
  dock.onclick = () => {
    if (memory.classList.contains('detached')) dockMemory();
    else undock();
  };
  let revealed = false;
  revealMemory = () => {
    if (revealed) return;
    revealed = true;
    memory.hidden = false;
    if (document.documentElement.clientWidth >= 1100) undock();
  };
  let drag = null;
  grip.onpointerdown = e => {
    if (e.button !== 0) return;
    undock();
    const rect = memory.getBoundingClientRect();
    drag = {x: e.clientX, y: e.clientY, left: rect.left, top: rect.top};
    grip.setPointerCapture(e.pointerId);
    e.preventDefault();
  };
  grip.onpointermove = e => {
    if (drag) place(drag.left + e.clientX - drag.x, drag.top + e.clientY - drag.y);
  };
  grip.onpointerup = grip.onpointercancel = grip.onlostpointercapture = () => { drag = null; };
  grip.onkeydown = e => {
    const movement = {ArrowLeft: [-16, 0], ArrowRight: [16, 0], ArrowUp: [0, -16], ArrowDown: [0, 16]}[e.key];
    if (!movement) return;
    e.preventDefault();
    undock();
    const rect = memory.getBoundingClientRect();
    place(rect.left + movement[0], rect.top + movement[1]);
  };
  addEventListener('resize', () => {
    if (!memory.classList.contains('detached')) return;
    if (document.documentElement.clientWidth < 1100) { dockMemory(); return; }
    const rect = memory.getBoundingClientRect();
    place(rect.left, rect.top);
  });
  updateDock();
  setupConsole.dockMemory = dockMemory;
  const completion = document.createElement('p');
  completion.id = 'completion';
  completion.hidden = true;
  completion.setAttribute('role', 'status');
  document.querySelector('.progress').after(completion);
  const flow = document.getElementById('flow');
  flow.classList.add('timeline');
  flow.innerHTML = '<div class="timeline-toolbar"><span id="stream-status">Ready</span><select id="event-source" aria-label="Timeline source"><option value="highlights">Task events</option><option value="all">All events</option></select></div><div class="timeline-legend"></div><div id="timeline-list" role="log" aria-label="Execution timeline" aria-live="off"></div>';
  for (const [key, source] of Object.entries(eventSources)) {
    const option = new Option(source.label, key);
    document.getElementById('event-source').add(option);
    const label = document.createElement('span');
    label.style.setProperty('--source', source.color);
    label.textContent = source.label;
    flow.querySelector('.timeline-legend').append(label);
  }
  document.getElementById('event-source').onchange = () => { timelineKey = ''; drawTimeline(); };
}

function renderDrawers(state) {
  const inspection = state.inspection;
  if (!inspection) return;
  const choice = document.getElementById('drawer-choice');
  if (!choice.options.length) {
    const list = document.getElementById('drawer-list');
    let group;
    for (const drawer of inspection.drawers) {
      if (!group || group.label !== drawer.group) {
        group = document.createElement('optgroup');
        group.label = drawer.group;
        choice.append(group);
        const heading = document.createElement('h3');
        heading.textContent = drawer.group;
        list.append(heading);
      }
      group.append(new Option(drawer.label, drawer.id));
      const row = document.createElement('div');
      row.className = 'drawer-row';
      row.dataset.drawer = drawer.id;
      const label = document.createElement('span');
      label.textContent = drawer.label;
      const status = document.createElement('small');
      status.className = 'drawer-state';
      row.append(label, status);
      for (const [fraction, action, icon] of [[1,'Open','chevrons-right'],[0,'Close','chevrons-left']]) {
        const button = document.createElement('button');
        button.className = 'icon';
        button.dataset.fraction = fraction;
        button.title = button.ariaLabel = `${action} ${drawer.group}: ${drawer.label}`;
        button.innerHTML = `<i data-lucide="${icon}"></i>`;
        button.onclick = () => { choice.value = drawer.id; command('drawer', {id:drawer.id,fraction}); };
        row.append(button);
      }
      list.append(row);
    }
    lucide.createIcons();
  }
  if (inspection.selected !== renderDrawers.selected) {
    if (inspection.selected) choice.value = inspection.selected;
    renderDrawers.selected = inspection.selected;
  }
  for (const drawer of inspection.drawers) {
    const row = document.querySelector(`[data-drawer="${drawer.id}"]`);
    row.classList.toggle('selected', drawer.id === choice.value);
    row.querySelector('.drawer-state').textContent = drawer.moving ? 'Moving' : drawer.fraction < .02 ? 'Closed' : Math.round(drawer.fraction * 100) + '%';
    for (const button of row.querySelectorAll('button')) button.disabled = !drawer.moving && Math.abs(drawer.fraction - Number(button.dataset.fraction)) < .01;
    if (drawer.id === choice.value && document.activeElement.id !== 'drawer-opening') {
      document.getElementById('drawer-opening').value = Math.round(drawer.fraction * 100);
      document.getElementById('drawer-percent').textContent = Math.round(drawer.fraction * 100) + '%';
    }
  }
  if (inspection.active) {
    document.getElementById('status').textContent = 'Inspection';
    document.getElementById('status').className = 'status';
    document.getElementById('step').disabled = true;
    document.getElementById('completion').hidden = true;
    document.getElementById('placement').textContent = 'Paused';
    document.getElementById('stream-status').textContent = 'Paused';
    if (document.getElementById('toggle').textContent.trim() !== 'Run again') {
      document.getElementById('toggle').innerHTML = '<i data-lucide="rotate-ccw"></i><span>Run again</span>';
      lucide.createIcons();
    }
  }
}

function isTaskEvent(event) {
  if (['memory', 'verification'].includes(event.source)) return true;
  if (event.source === 'dispatch') return /^(Open drawer|Inspect drawer|Grasp seasoning|Lift seasoning|Lower seasoning|Release seasoning|Close drawer|Verify retrieval)/.test(event.message);
  return event.source === 'simulation' && /^(Stopped:|Completed: Close drawer)/.test(event.message);
}

function drawTimeline() {
  const filter = document.getElementById('event-source').value;
  const key = filter + ':' + timelineEvents.map(e => e.id).join(',');
  if (key === timelineKey) return;
  timelineKey = key;
  const list = document.getElementById('timeline-list');
  const oldTop = list.scrollTop, oldHeight = list.scrollHeight;
  const events = timelineEvents.filter(e => filter === 'all' || (filter === 'highlights' ? isTaskEvent(e) : e.source === filter)).slice(-80).reverse();
  list.replaceChildren(...events.map(event => {
    const source = eventSources[event.source];
    const row = document.createElement('article');
    row.className = 'timeline-event' + (event.id > lastEventId ? ' arriving' : '');
    row.style.setProperty('--source', source.color);
    const meta = document.createElement('div');
    meta.className = 'timeline-meta';
    const label = document.createElement('strong');
    label.textContent = source.label;
    const time = document.createElement('time');
    time.textContent = event.time.toFixed(2) + ' s';
    meta.append(label, time);
    const message = document.createElement('p');
    message.textContent = event.message;
    row.append(meta, message);
    return row;
  }));
  if (!events.length) {
    const empty = document.createElement('p');
    empty.className = 'timeline-empty';
    empty.textContent = ['all', 'highlights'].includes(filter) ? 'Waiting for execution' : 'No events from this flow';
    list.append(empty);
  }
  list.scrollTop = oldTop > 12 ? oldTop + list.scrollHeight - oldHeight : 0;
  lastEventId = Math.max(lastEventId, ...timelineEvents.map(e => e.id));
}

function renderConsole(state, live) {
  if (state.task === 'Drawer search') revealMemory();
  document.getElementById('camera-tour').setAttribute('aria-pressed', String(state.camera_tour));
  if (state.camera_tour && !renderConsole.wasTour) setupConsole.dockMemory();
  renderConsole.wasTour = state.camera_tour;
  if (state.camera_tour) document.querySelectorAll('[data-camera]').forEach(button => button.setAttribute('aria-pressed', 'false'));
  document.getElementById('playback-speed').textContent = state.camera_opening ? 'Overview' : state.speed + 'x';
  document.querySelector('.playback-badge small').textContent = state.camera_opening ? 'CAMERA OPENING' : 'SIMULATION SPEED';
  const motion = state.motion;
  document.getElementById('flow').classList.toggle('live', live);
  document.getElementById('stream-status').textContent = state.camera_opening ? 'Opening view' : motion.finished ? 'Run complete' : live ? 'Streaming' : 'Ready';
  if (renderConsole.cycle !== state.run_cycle) {
    timelineKey = '';
    lastEventId = 0;
    document.getElementById('timeline-list').scrollTop = 0;
    renderConsole.cycle = state.run_cycle;
  }
  timelineEvents = state.timeline || [];
  drawTimeline();
  const completion = document.getElementById('completion');
  completion.hidden = !(state.task === 'Drawer search' && motion.finished && motion.success);
  completion.textContent = 'Seasoning placed on counter. All drawers closed.';
  document.getElementById('drawers-checked').textContent = `${motion.observations?.length || 0} / 4`;
  const knownDrawer = ['top left', 'top right', 'bottom left', 'bottom right'].indexOf(motion.target_drawer) + 1;
  document.getElementById('object-location').textContent = motion.object_placed ? 'On counter' : motion.found ? 'Found' : motion.memory_used ? `Drawer ${knownDrawer}` : 'Unknown';
  for (const row of ['top', 'bottom']) for (const side of ['left', 'right']) {
    const cell = document.getElementById(`memory-${row}-${side}`).parentElement;
    const value = motion.memory?.[row + ' ' + side] || '';
    if (cell.dataset.value !== undefined && cell.dataset.value !== value && value) {
      cell.classList.remove('memory-updated');
      void cell.offsetWidth;
      cell.classList.add('memory-updated');
    }
    cell.dataset.value = value;
    cell.classList.toggle('remembered', Boolean(value));
    const found = Boolean(motion.found && motion.target_drawer === row + ' ' + side);
    cell.classList.toggle('target-found', found);
    let badge = cell.querySelector('.found-badge');
    if (found && !badge) {
      badge = document.createElement('strong');
      badge.className = 'found-badge';
      cell.append(badge);
    }
    if (badge) {
      badge.hidden = !found;
      badge.textContent = motion.object_placed ? 'On counter' : 'Found';
    }
  }
}
