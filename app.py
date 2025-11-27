import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from color_engine import auto_color_single_image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# static 아래에 두어서 바로 브라우저에서 볼 수 있게 함
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result_image_url = None
    original_image_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "파일을 선택해 주세요."
        else:
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)

            # 원본 이미지 미리보기용 URL
            original_image_url = url_for(
                "static", filename=f"uploads/{filename}"
            )

            palette = request.form.get("palette", "default")
            use_guide = bool(request.form.get("use_guide"))

            result_name = f"result_{filename}"
            output_path = os.path.join(RESULT_FOLDER, result_name)

            try:
                auto_color_single_image(
                    input_path,
                    output_path,
                    palette_name=palette,
                    use_guide=use_guide,
                )
                result_image_url = url_for(
                    "static", filename=f"results/{result_name}"
                )
            except Exception as e:
                print("[ERROR] 이미지 처리 중 예외:", e)
                error = "이미지 처리 중 오류가 발생했어요."

    return render_template(
        "index.html",
        result_image=result_image_url,
        original_image=original_image_url,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
