<!-- Save as form.php -->
<!DOCTYPE html>
<html>
<head>
    <title>Text Formatter</title>
</head>
<body>

<form action="display.php" method="post">
    <label>Enter your text:</label><br>
    <textarea name="user_text" rows="5" cols="40"></textarea><br><br>

    <label>Select text color:</label>
    <input type="color" name="color"><br><br>

    <label>Select font:</label>
    <select name="font">
        <option value="Arial">Arial</option>
        <option value="Georgia">Georgia</option>
        <option value="Courier New">Courier New</option>
        <option value="Verdana">Verdana</option>
    </select><br><br>

    <label>Font size (px):</label>
    <input type="number" name="size" min="10" max="72"><br><br>

    <label>Save preferences?</label>
    <input type="radio" name="save" value="yes"> Yes
    <input type="radio" name="save" value="no" checked> No<br><br>

    <input type="submit" value="Submit">
</form>

</body>
</html>
