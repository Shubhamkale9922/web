<!DOCTYPE html>
<html>
<head>
    <title>Book Details Form</title>
</head>
<body>

<h2>Enter Book Details</h2>

<form action="submit_book.php" method="post">
    <label>Book Name:</label><br>
    <input type="text" name="book_name" required><br><br>

    <label>Author Name:</label><br>
    <input type="text" name="author_name" required><br><br>

    <label>Publisher Name:</label><br>
    <input type="text" name="publisher_name" required><br><br>

    <label>Category:</label><br>
    <input type="text" name="category" required><br><br>

    <label>Synopsis:</label><br>
    <textarea name="synopsis" rows="5" cols="40" required></textarea><br><br>

    <input type="submit" value="Submit">
    <input type="reset" value="Reset">
</form>

</body>
</html>
