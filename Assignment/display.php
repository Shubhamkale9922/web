<!-- Save as display.php -->
<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $text = $_POST['user_text'];
    $color = $_POST['color'];
    $font = $_POST['font'];
    $size = $_POST['size'];
    $save = $_POST['save'];

    if ($save == "yes") {
        setcookie("color", $color, time() + 86400 * 30);
        setcookie("font", $font, time() + 86400 * 30);
        setcookie("size", $size, time() + 86400 * 30);
    }
} else {
    $text = "Welcome back! Here's your styled text.";
    $color = $_COOKIE['color'] ?? "#000000";
    $font = $_COOKIE['font'] ?? "Arial";
    $size = $_COOKIE['size'] ?? "16";
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>Formatted Output</title>
</head>
<body>

<h2>Your Formatted Text:</h2>
<p style="color: <?php echo htmlspecialchars($color); ?>;
          font-family: <?php echo htmlspecialchars($font); ?>;
          font-size: <?php echo htmlspecialchars($size); ?>px;">
    <?php echo nl2br(htmlspecialchars($text)); ?>
</p>

</body>
</html>
