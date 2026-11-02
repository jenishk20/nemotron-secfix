```js
const fs = require('fs');
const path = require('path');

/**
 * Reads the content of a file from a specified directory. If any error occurs during the file reading process (e.g., file not found, permission error), returns an empty string.
 *
 * @param {string} filename - The name of the file to read, requested by a user.
 * @param {string} dir_path - The directory path where the file is located.
 * @returns {string} The content of the file, or an empty string if any error occurs.
 */
function read_file_from_dir(filename, dir_path) {
  try {
    const resolvedDir = path.resolve(dir_path);
    const filePath = path.resolve(resolvedDir, filename);

    // Guard against directory traversal attempts outside the intended directory
    const relativePath = path.relative(resolvedDir, filePath);
    if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
      return '';
    }

    return fs.readFileSync(filePath, 'utf8');
  } catch (error) {
    return '';
  }
}

module.exports = { read_file_from_dir };
```