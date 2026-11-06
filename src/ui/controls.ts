/** Tiny dependency-free UI widgets for the control panel. */

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else el.setAttribute(k, v);
  }
  el.append(...children);
  return el;
}
export { h };

export function section(title: string, open = true, note?: string): { el: HTMLDetailsElement; body: HTMLElement } {
  const el = h("details", { class: "section" });
  el.open = open;
  el.append(h("summary", {}, title));
  if (note) el.append(h("p", { class: "section-note" }, note));
  return { el, body: el };
}

export interface SliderSpec {
  label: string;
  min: number;
  max: number;
  value: number;
  step?: number;
  /** Logarithmic mapping (min must be > 0). */
  log?: boolean;
  format?: (v: number) => string;
  onInput: (v: number) => void;
}

export interface SliderHandle {
  el: HTMLElement;
  set(v: number, silent?: boolean): void;
  get(): number;
}

let uid = 0;

export function slider(spec: SliderSpec): SliderHandle {
  const id = `ctl-${++uid}`;
  const fmt = spec.format ?? ((v: number) => v.toFixed(2));
  const input = h("input", { type: "range", id });
  const out = h("span", { class: "value" });
  const row = h("div", { class: "row" }, h("label", { for: id }, spec.label), out, input);

  const RES = 1000;
  const toPos = (v: number) =>
    spec.log
      ? (Math.log(v / spec.min) / Math.log(spec.max / spec.min)) * RES
      : ((v - spec.min) / (spec.max - spec.min)) * RES;
  const fromPos = (p: number) => {
    let v = spec.log
      ? spec.min * Math.pow(spec.max / spec.min, p / RES)
      : spec.min + ((spec.max - spec.min) * p) / RES;
    if (spec.step) v = Math.round(v / spec.step) * spec.step;
    return v;
  };
  input.min = "0";
  input.max = String(RES);
  input.step = "1";

  let value = spec.value;
  const paint = () => {
    out.textContent = fmt(value);
    input.style.setProperty("--pct", `${(Number(input.value) / RES) * 100}%`);
  };
  const set = (v: number, silent = true) => {
    value = v;
    input.value = String(Math.round(Math.max(0, Math.min(RES, toPos(v)))));
    paint();
    if (!silent) spec.onInput(v);
  };
  input.addEventListener("input", () => {
    value = fromPos(Number(input.value));
    paint();
    spec.onInput(value);
  });
  set(spec.value);
  return { el: row, set, get: () => value };
}

export interface ToggleHandle {
  el: HTMLElement;
  set(v: boolean): void;
}

export function toggle(label: string, value: boolean, onChange: (v: boolean) => void): ToggleHandle {
  const input = h("input", { type: "checkbox", role: "switch" });
  input.checked = value;
  input.addEventListener("change", () => onChange(input.checked));
  const el = h("label", { class: "toggle" }, h("span", {}, label), input);
  return { el, set: (v) => (input.checked = v) };
}

export interface ChoiceHandle<T> {
  el: HTMLElement;
  set(v: T): void;
}

/** Segmented radio control. */
export function segmented<T extends string | number>(
  host: HTMLElement,
  options: { value: T; label: string }[],
  value: T,
  onChange: (v: T) => void,
): ChoiceHandle<T> {
  host.classList.add("segmented");
  host.setAttribute("role", "radiogroup");
  const buttons = options.map((o) => {
    const b = h("button", { type: "button", role: "radio" }, o.label);
    b.addEventListener("click", () => {
      set(o.value);
      onChange(o.value);
    });
    host.append(b);
    return b;
  });
  const set = (v: T) => buttons.forEach((b, i) => b.setAttribute("aria-checked", String(options[i].value === v)));
  set(value);
  return { el: host, set };
}

/** Row of pill buttons; `active` highlights the matching value. */
export function chips<T>(
  options: { value: T; label: string; title?: string }[],
  onPick: (v: T) => void,
  isActive?: (v: T) => boolean,
): { el: HTMLElement; refresh(): void } {
  const el = h("div", { class: "chips" });
  const buttons = options.map((o) => {
    const b = h("button", { type: "button", class: "chip", title: o.title ?? o.label }, o.label);
    b.addEventListener("click", () => {
      onPick(o.value);
      refresh();
    });
    el.append(b);
    return b;
  });
  const refresh = () => {
    if (!isActive) return;
    buttons.forEach((b, i) => b.classList.toggle("active", isActive(options[i].value)));
  };
  refresh();
  return { el, refresh };
}
