const express = require("express");
const path = require("path");

const app = express();

// index: false — otherwise express.static would answer GET "/" with public/index.html
// itself and the "/" route below could never run.
app.use(express.static(path.join(__dirname, "public"), { index: false }));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

const PORT = process.env.PORT || 3000;

// Only listen when run directly (`node server.js`) so the tests can import the app.
if (require.main === module) {
  app.listen(PORT, () => console.log(`Listening on http://localhost:${PORT}`));
}

module.exports = app;
