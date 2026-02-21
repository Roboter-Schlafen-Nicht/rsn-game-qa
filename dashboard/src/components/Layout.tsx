import { NavLink, Outlet } from "react-router-dom";
import {
  LayoutDashboard,
  GraduationCap,
  ClipboardCheck,
  AlertTriangle,
  GitCompare,
  Bug,
} from "lucide-react";

const NAV_ITEMS = [
  { to: "/overview", label: "Overview", icon: LayoutDashboard },
  { to: "/training", label: "Training", icon: GraduationCap },
  { to: "/evaluation", label: "Evaluation", icon: ClipboardCheck },
  { to: "/findings", label: "Findings", icon: AlertTriangle },
  { to: "/cross-game", label: "Cross-Game", icon: GitCompare },
];

export function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="flex w-60 flex-col border-r border-zinc-800 bg-zinc-950">
        {/* Logo */}
        <div className="flex h-14 items-center gap-2.5 border-b border-zinc-800 px-5">
          <Bug className="h-5 w-5 text-brand-500" />
          <span className="text-sm font-semibold tracking-tight">
            RSN Game QA
          </span>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-0.5 px-3 py-3">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-zinc-800/80 text-zinc-100"
                    : "text-zinc-400 hover:bg-zinc-800/40 hover:text-zinc-200"
                }`
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t border-zinc-800 px-5 py-3">
          <p className="text-[10px] text-zinc-600">
            v0.1.0-alpha &middot; 1137 tests
          </p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-zinc-950">
        <div className="mx-auto max-w-7xl px-6 py-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
