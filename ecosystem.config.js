module.exports = {
  apps: [
    {
      name: "api-server",
      script: "api_server.py",
      interpreter: "python3.10",
      watch: false,
      env: {
        PORT: 8000,
      },
    },
    // Run Python HTTP server as a module, matching run.sh logic
    {
      name: "http-server",
      script: "python3",
      args: "-m http.server 8005",
      watch: false,
    },
  ],
};
