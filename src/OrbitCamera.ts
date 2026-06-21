export class OrbitCamera {
  yaw = 0;
  pitch = -0.18;
  zoom = 1;
  panX = 0;
  panY = 0;
  projectedX = 0;
  projectedY = 0;
  projectedScale = 1;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private cosYaw = 1;
  private sinYaw = 0;
  private cosPitch = 1;
  private sinPitch = 0;
  private focal = 1;

  constructor(private readonly canvas: HTMLCanvasElement) {
    canvas.addEventListener("pointerdown", event => {
      this.dragging = true; this.lastX = event.clientX; this.lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("dragging");
    });
    canvas.addEventListener("pointermove", event => {
      if (!this.dragging) return;
      this.yaw += (event.clientX - this.lastX) * .006;
      this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + (event.clientY - this.lastY) * .006));
      this.lastX = event.clientX; this.lastY = event.clientY;
    });
    const stop = (event: PointerEvent): void => {
      this.dragging = false; canvas.classList.remove("dragging");
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      const unit = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 16 : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? innerHeight : 1;
      if (event.ctrlKey) {
        // Browsers expose a touchpad pinch as a wheel event with ctrlKey set.
        this.zoom = Math.max(.03, Math.min(50, this.zoom * Math.exp(-event.deltaY * unit * .01)));
      } else {
        const dpr = Math.min(devicePixelRatio, 1.5);
        const horizontal = event.shiftKey && event.deltaX === 0 ? event.deltaY : event.deltaX;
        const vertical = event.shiftKey && event.deltaX === 0 ? 0 : event.deltaY;
        this.panX -= horizontal * unit * dpr;
        this.panY -= vertical * unit * dpr;
      }
    }, { passive: false });
  }

  prepare(): void {
    this.cosYaw = Math.cos(this.yaw); this.sinYaw = Math.sin(this.yaw);
    this.cosPitch = Math.cos(this.pitch); this.sinPitch = Math.sin(this.pitch);
    this.focal = Math.max(this.canvas.width, this.canvas.height) * 1.15;
  }

  project(x: number, y: number, z: number, centerX: number, centerY: number, centerZ = 0): boolean {
    const dx = x - centerX, dy = y - centerY, dz = z - centerZ;
    const rx = this.cosYaw * dx - this.sinYaw * dz;
    const rz = this.sinYaw * dx + this.cosYaw * dz;
    const ry = this.cosPitch * dy - this.sinPitch * rz;
    const depth = this.sinPitch * dy + this.cosPitch * rz;
    const denominator = this.focal + depth * this.zoom;
    if (denominator < 20) return false;
    this.projectedScale = this.focal / denominator * this.zoom;
    this.projectedX = this.canvas.width / 2 + this.panX + rx * this.projectedScale;
    this.projectedY = this.canvas.height / 2 + this.panY + ry * this.projectedScale;
    return true;
  }
}
