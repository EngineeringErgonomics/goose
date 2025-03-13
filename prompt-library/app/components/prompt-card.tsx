import { Link } from "react-router";
import { Puzzle } from "lucide-react";
import type { Prompt } from "../types/prompt";

interface PromptCardProps {
  prompt: Prompt;
}

export function PromptCard({ prompt }: PromptCardProps) {
  // Function to get extension name from extension string
  const getExtensionName = (extension: string) => {
    // Map the extension strings to the display names we want
    const extensionMap: { [key: string]: string } = {
      'developer': 'developer',
      'computercontroller': 'computer controller',
      'memory': 'memory',
      'jetbrains': 'jetbrains',
      'git': 'git',
      'figma': 'figma'
    };
    
    const parts = extension.split('__');
    return extensionMap[parts[0]] || parts[0];
  };

  // Get unique extensions
  const extensions = [...new Set(prompt.extensions.map(getExtensionName))];

  return (
    <div className="relative h-full p-[2px] overflow-hidden rounded-[17px] group/card bg-borderSubtle hover:bg-transparent hover:duration-300">
      <div className="absolute opacity-0 group-hover/card:opacity-100 group-hover/card:duration-200 pointer-events-none w-[600px] h-[600px] top-[-150px] left-[-50px] origin-center bg-[linear-gradient(45deg,#13BBAF,#FF4F00)] animate-[rotate_6s_linear_infinite] z-[-1]"></div>
      <Link
        to={`/detail/${prompt.id}`}
        className="block p-6 rounded-[15px] bg-white dark:bg-black h-[300px] flex flex-col"
      >
        <div className="mb-3">
          <h3 className="text-lg font-medium text-textProminent line-clamp-1">{prompt.title}</h3>
          <p className="text-sm text-textSubtle mt-1 line-clamp-2">{prompt.description}</p>
        </div>

        {/* Placeholder Image */}
        <div className="h-32 mb-3 rounded-md overflow-hidden bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-900">
          <div className="w-full h-full flex items-center justify-center text-gray-400 text-sm">
            Preview Coming Soon
          </div>
        </div>

        <div className="mt-auto pt-4 border-t border-borderSubtle">
          {extensions.length > 0 && (
            <div className="flex items-center gap-2">
              <Puzzle className="h-4 w-4 text-textSubtle" />
              <div className="text-sm text-textSubtle">
                {extensions.join(", ")}
              </div>
            </div>
          )}
        </div>
      </Link>
    </div>
  );
}