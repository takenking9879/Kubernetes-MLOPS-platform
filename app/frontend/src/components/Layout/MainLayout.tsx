import { Header } from './Header';
import { CatalogPanel } from '../Catalog/CatalogPanel';
import { CanvasPanel } from '../Canvas/CanvasPanel';
import { PropertiesPanel } from '../Properties/PropertiesPanel';
import { ConsolePanel } from '../Console/ConsolePanel';

export function MainLayout() {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-slate-950 text-slate-100">
      {/* Top bar */}
      <Header />

      {/* Main content area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Catalog */}
        <CatalogPanel />

        {/* Center + Bottom */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Canvas */}
          <div className="flex-1 overflow-hidden">
            <CanvasPanel />
          </div>

          {/* Console */}
          <div className="h-36 shrink-0">
            <ConsolePanel />
          </div>
        </div>

        {/* Right: Properties */}
        <PropertiesPanel />
      </div>
    </div>
  );
}
