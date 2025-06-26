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
    {
      name: "http-server",
      script: "http.server",
      interpreter: "python3",
      interpreter_args: "-m",
      args: "8005",
      watch: false,
    },
  ],
};
