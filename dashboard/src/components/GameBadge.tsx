import { gameDisplayName, gameColor } from "@/lib/utils";

interface GameBadgeProps {
  game: string;
}

export function GameBadge({ game }: GameBadgeProps) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md bg-zinc-800/50 border border-zinc-700/50 px-2 py-0.5 text-xs font-medium ${gameColor(game)}`}>
      {gameDisplayName(game)}
    </span>
  );
}
