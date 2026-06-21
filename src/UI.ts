import { CONTROL_DEFINITIONS, type SimulationConfig } from "./config.js";

export class UI {
  private readonly pause = document.querySelector<HTMLButtonElement>("#pause")!;

  constructor(config: SimulationConfig, onChange: (next: SimulationConfig) => void) {
    const controls = document.querySelector<HTMLElement>("#controls")!;
    for (const definition of CONTROL_DEFINITIONS) {
      const row = document.createElement("label");
      row.className = "control";
      row.innerHTML = `<span>${definition.label}</span><output>${definition.format(config[definition.key])}</output><input type="range" min="${definition.min}" max="${definition.max}" step="${definition.step}" value="${config[definition.key]}" />`;
      const input = row.querySelector<HTMLInputElement>("input")!;
      const output = row.querySelector<HTMLOutputElement>("output")!;
      input.addEventListener("input", () => {
        config[definition.key] = Number(input.value);
        output.value = definition.format(config[definition.key]);
        onChange({ ...config });
      });
      controls.append(row);
    }
  }

  update(paused: boolean): void {
    this.pause.textContent = paused ? "▶" : "Ⅱ";
  }
}
