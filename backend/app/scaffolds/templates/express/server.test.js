const test = require("node:test");
const assert = require("node:assert");
const http = require("node:http");

const app = require("./server");

test("GET / responds with 200", async () => {
  const server = app.listen(0);
  try {
    const { port } = server.address();
    const status = await new Promise((resolve, reject) => {
      http
        .get({ port, path: "/" }, (res) => {
          res.resume();
          resolve(res.statusCode);
        })
        .on("error", reject);
    });
    assert.strictEqual(status, 200);
  } finally {
    server.close();
  }
});
