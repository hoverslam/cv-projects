import numpy as np
import cv2

from sudoku import Sudoku as SudokuSolver

from pytesseract import image_to_string
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border


class SudokuImage:
    """A standard 9x9 Sudoku puzzle represented by an image. The board can be extracted from the image.
    """

    def __init__(self, file: str) -> None:
        """Initialize the image.

        Args:
            file (str): Path to the image file.
        """
        self.img = self.resize_height(cv2.imread(file), 500)

    def solve(self) -> None:
        """Solve puzzle and display the original board with the missing digits.
        """
        # Find solution
        board = self.get_board()
        digits = self.get_digits()
        solver = SudokuSolver(3, 3, board=digits.tolist())
        solution = np.array(solver.solve().board)

        # Divide board in its 81 cells
        height, width = board.shape[:2]
        h_linspace = np.linspace(0, height, 10, dtype=int)
        w_linspace = np.linspace(0, width, 10, dtype=int)

        # Put solution in cells
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize("0", font, 0.8, 2)[0]
        for i in range(9):
            for j in range(9):
                cell = board[h_linspace[i]:h_linspace[i+1], w_linspace[j]:w_linspace[j+1]]
                textX = (cell.shape[1] - textsize[0]) // 2 + w_linspace[j]
                textY = (cell.shape[0] + textsize[1]) // 2 + h_linspace[i]
                if digits[i, j] == 0:
                    cv2.putText(board, str(solution[i, j]), (textX, textY), font, 0.8, (0, 0, 255), 2)

        cv2.imshow("Solution", board)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def resize_height(self, img: np.ndarray, height: int) -> np.ndarray:
        """Resize image to a given height but keep aspect ratio.

        Args:
            img (np.ndarray): The image to resize.
            height (int): The new height of the image.

        Returns:
            np.ndarray: The resized image.
        """
        original_width, original_height = img.shape[:2]
        scale_factor = height / original_height
        width = int(scale_factor * original_width)

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def get_border(self, debug: bool = False) -> np.ndarray:
        """Find the border of the Sudoku board.

        Args:
            debug (bool, optional): Show image with board border. Defaults to False.

        Returns:
            np.ndarray: The contour of the border.
        """
        # Preprocess image to find the border easier
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 4)

        # Find the border by assuming the Sudoku board is the 2nd largest area
        # TODO: Find a better way to extract the border since this crude assumption will not hold in most cases.
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        border = sorted(contours, key=cv2.contourArea, reverse=True)[1]

        if debug:
            cv2.drawContours(self.img, [border], 0, (0, 255, 0), 1)
            cv2.imshow("Sudoku", self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return border

    def get_board(self) -> np.ndarray:
        """Create an top-down image of the board only.

        Returns:
            np.ndarray: The puzzle board.
        """
        # Find the four corners of the puzzle
        border = self.get_border()
        peri = cv2.arcLength(border, True)
        corners = cv2.approxPolyDP(border, 0.04 * peri, True)
        corners = np.squeeze(corners, axis=1)

        # Deskew to get a top-down view of the board
        board = four_point_transform(self.img, corners)

        return board

    def get_digits(self) -> np.ndarray:
        """Extract the digits from the puzzle board.

        Returns:
            np.ndarray: The digits as an 9x9 array. Zero means empty cell.
        """
        # Preprocessing
        board = cv2.resize(self.get_board(), (800, 800))
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        borderless = clear_border(thresh)
        erosion = cv2.erode(borderless, (5, 5), iterations=2)

        # Divide board in its 81 cells
        height, width = board.shape[:2]
        h_linspace = np.linspace(0, height, 10, dtype=int)
        w_linspace = np.linspace(0, width, 10, dtype=int)

        # Loop over all cells and extract the digits
        # TODO: Train a model on MNIST digits (maybe)
        digits = np.zeros((9, 9), dtype=int)
        for i in range(9):
            for j in range(9):
                cell = erosion[h_linspace[i]:h_linspace[i+1], w_linspace[j]:w_linspace[j+1]]
                ocr = image_to_string(cell, config="--oem 3 --psm 10").strip()
                if ocr.isnumeric():
                    digits[i, j] = int(ocr)

        return digits


if __name__ == "__main__":
    sudoku = SudokuImage("img/sudoku1.jpg")
    sudoku.solve()

    # board = np.array([
    #     [8, 0, 0, 0, 1, 0, 0, 0, 9],
    #     [0, 5, 0, 8, 0, 7, 0, 1, 0],
    #     [0, 0, 4, 0, 9, 0, 7, 0, 0],
    #     [0, 6, 0, 7, 0, 1, 0, 2, 0],
    #     [5, 0, 8, 0, 6, 0, 1, 0, 7],
    #     [0, 1, 0, 5, 0, 2, 0, 9, 0],
    #     [0, 0, 7, 0, 4, 0, 6, 0, 0],
    #     [0, 8, 0, 3, 0, 9, 0, 4, 0],
    #     [3, 0, 0, 0, 5, 0, 0, 0, 8]
    # ], dtype=int)

    # print(board == sudoku.get_digits())