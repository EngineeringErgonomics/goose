import { Search } from "lucide-react";
import { Button } from "@site/src/components/ui/button";
import { Input } from "@site/src/components/ui/input";
import { ServerCard } from "@site/src/components/server-card";
import { useState, useEffect } from "react";
import type { MCPServer } from "@site/src/types/server";
import { fetchMCPServers, searchMCPServers } from "@site/src/utils/mcp-servers";
import { motion } from "framer-motion";
import Layout from "@theme/Layout";

export default function HomePage() {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Combined effect for initial load and search
  useEffect(() => {
    const loadServers = async () => {
      try {
        setIsLoading(true);
        setError(null);

        const trimmedQuery = searchQuery.trim();
        const results = trimmedQuery
          ? await searchMCPServers(trimmedQuery)
          : await fetchMCPServers();

        console.log('Loaded servers:', results);
        setServers(results);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Unknown error";
        setError(`Failed to load servers: ${errorMessage}`);
        console.error("Error loading servers:", err);
      } finally {
        setIsLoading(false);
      }
    };

    // Debounce all server loads
    const timeoutId = setTimeout(loadServers, 300);
    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  return (
    <Layout>
      <div className="container mx-auto px-4 pb-24">
        <div className="pb-16">
          <h1 className="text-[64px] font-medium">Browse Extensions</h1>
          <p>Your central directory for discovering and installing extensions.</p>
        </div>

        <div className="search-container">
          <input
            className="search-input"
            placeholder="Search for extensions"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        {error && (
          <div className="p-4 bg-red-50 text-red-600 rounded-md">{error}</div>
        )}

        <section className="pt-8">
          <div className={`${searchQuery ? "pb-2" : "pb-8"}`}>
            <p className="text-gray-600">
              {searchQuery
                ? `${servers.length} result${
                    servers.length > 1 ? "s" : ""
                  } for "${searchQuery}"`
                : ""}
            </p>
          </div>

          {isLoading ? (
            <div className="py-8 text-xl text-gray-600">Loading servers...</div>
          ) : servers.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              {searchQuery
                ? "No servers found matching your search."
                : "No servers available."}
            </div>
          ) : (
            <div className="cards-grid">
              {servers
                .sort((a, b) => {
                  // Sort built-in servers first
                  if (a.is_builtin && !b.is_builtin) return -1;
                  if (!a.is_builtin && b.is_builtin) return 1;
                  return 0;
                })
                .map((server) => (
                  <motion.div
                    key={server.id}
                    initial={{
                      opacity: 0,
                    }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.6 }}
                  >
                    <ServerCard key={server.id} server={server} />
                  </motion.div>
                ))}
            </div>
          )}
        </section>
      </div>
    </Layout>
  );
}