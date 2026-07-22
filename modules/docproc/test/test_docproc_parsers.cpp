// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "../src/doctags_parser.hpp"
#include "../src/markdown_parser.hpp"
#include "../src/plain_text_parser.hpp"

namespace opencv_test { namespace {

using namespace cv::docproc;

TEST(Docproc_DocTagsParser, IsDocTags)
{
    EXPECT_TRUE(isDocTags("<doctag><text>hi</text></doctag>"));
    EXPECT_FALSE(isDocTags("plain text"));
}

TEST(Docproc_DocTagsParser, TextTagBecomesTextBlock)
{
    Page page;
    parseDocTagsPage("<doctag><text>Hello world</text></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    EXPECT_EQ(BLOCK_TEXT, page.blocks[0].type);
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size());
    EXPECT_EQ("Hello world", page.blocks[0].lines[0].text);
}

TEST(Docproc_DocTagsParser, HeaderTagBecomesTitleBlock)
{
    Page page;
    parseDocTagsPage(
        "<doctag><section_header_level_1>Title Text</section_header_level_1></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    EXPECT_EQ(BLOCK_TITLE, page.blocks[0].type);
    EXPECT_EQ("Title Text", page.blocks[0].lines[0].text);
}

TEST(Docproc_DocTagsParser, RaggedTableColsMatchesWidestRow)
{
    Page page;
    parseDocTagsPage(
        "<doctag><otsl><fcel>A<fcel>B<nl><fcel>C<fcel>D<fcel>E<nl></otsl></doctag>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(2, block.rows);
    EXPECT_EQ(3, block.cols); // widest row (3 cells), not the first row (2 cells)
    ASSERT_EQ((size_t)5, block.cells.size());
}

TEST(Docproc_DocTagsParser, IsOtslCellStream)
{
    EXPECT_TRUE(isOtslCellStream("<fcel>A<fcel>B<nl>"));
    EXPECT_FALSE(isOtslCellStream("<doctag><text>hi</text></doctag>"));
}

TEST(Docproc_DocTagsParser, ParseOtslCellStreamHasNoClosingTag)
{
    Page page;
    parseOtslCellStream("<fcel>A<fcel>B<nl>", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(1, block.rows);
    EXPECT_EQ(2, block.cols);
    ASSERT_EQ((size_t)2, block.cells.size());
    EXPECT_EQ("A", block.cells[0].text);
    EXPECT_EQ("B", block.cells[1].text);
}

TEST(Docproc_MarkdownParser, IsMarkdown)
{
    EXPECT_TRUE(isMarkdown("# Title\n"));
    EXPECT_TRUE(isMarkdown("| A | B |\n| --- | --- |\n| 1 | 2 |\n"));
    EXPECT_FALSE(isMarkdown("Just plain text.\nAnother line.\n"));
}

TEST(Docproc_MarkdownParser, HeaderAndParagraph)
{
    Page page;
    parseMarkdownPage("# Title\n\nSome paragraph text.", page);

    ASSERT_EQ((size_t)2, page.blocks.size());
    EXPECT_EQ(BLOCK_TITLE, page.blocks[0].type);
    EXPECT_EQ("Title", page.blocks[0].lines[0].text);
    EXPECT_EQ(BLOCK_TEXT, page.blocks[1].type);
    EXPECT_EQ("Some paragraph text.", page.blocks[1].lines[0].text);
}

TEST(Docproc_MarkdownParser, PipeTable)
{
    Page page;
    parseMarkdownPage("| A | B |\n| --- | --- |\n| 1 | 2 |", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(2, block.rows);
    EXPECT_EQ(2, block.cols);
    ASSERT_EQ((size_t)4, block.cells.size());
    EXPECT_TRUE(block.cells[0].is_header);
    EXPECT_FALSE(block.cells[2].is_header);
    EXPECT_EQ("1", block.cells[2].text);
}

TEST(Docproc_PlainTextParser, SingleSpaceStaysInOneColumn)
{
    Page page;
    parsePlainTextPage("Jane Miller works here.", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    EXPECT_EQ(BLOCK_TEXT, page.blocks[0].type);
    ASSERT_EQ((size_t)1, page.blocks[0].lines.size());
    EXPECT_EQ("Jane Miller works here.", page.blocks[0].lines[0].text);
}

TEST(Docproc_PlainTextParser, WhitespaceAlignedColumnsBecomeTable)
{
    Page page;
    parsePlainTextPage("Name         Age\nJane Miller  30\nJohn Smith   25", page);

    ASSERT_EQ((size_t)1, page.blocks.size());
    const Block& block = page.blocks[0];
    EXPECT_EQ(BLOCK_TABLE, block.type);
    EXPECT_EQ(3, block.rows);
    EXPECT_EQ(2, block.cols);
    ASSERT_EQ((size_t)6, block.cells.size());
    EXPECT_EQ("Jane Miller", block.cells[2].text); // row 1, col 0 -- not split on the single space
    EXPECT_FALSE(block.cells[2].is_header);
}

TEST(Docproc_Page, GetWordsCollectsFromTextBearingBlocksOnly)
{
    Page page;

    Word w1; w1.text = "Hello";
    Line titleLine; titleLine.words = {w1};
    Block title; title.type = BLOCK_TITLE; title.lines = {titleLine};

    Word w2; w2.text = "Some";
    Word w3; w3.text = "text";
    Line textLine; textLine.words = {w2, w3};
    Block text; text.type = BLOCK_TEXT; text.lines = {textLine};

    Cell cell; cell.text = "1"; // table cells carry no Word breakdown
    Block table; table.type = BLOCK_TABLE; table.cells = {cell};

    page.blocks = {title, text, table};

    std::vector<Word> words = page.getWords();
    ASSERT_EQ((size_t)3, words.size());
    EXPECT_EQ("Hello", words[0].text);
    EXPECT_EQ("Some", words[1].text);
    EXPECT_EQ("text", words[2].text);
}

TEST(Docproc_Page, GetWordsSynthesizesFromLineTextWhenWordsEmpty)
{
    Page page;
    Line line; line.text = "Jane Miller works here.";
    Block text; text.type = BLOCK_TEXT; text.lines = {line};
    page.blocks = {text};

    std::vector<Word> words = page.getWords();
    ASSERT_EQ((size_t)4, words.size());
    EXPECT_EQ("Jane", words[0].text);
    EXPECT_EQ("Miller", words[1].text);
    EXPECT_EQ("works", words[2].text);
    EXPECT_EQ("here.", words[3].text);
}

TEST(Docproc_Page, GetTablesReturnsOnlyTableBlocksInOrder)
{
    Page page;
    Block text; text.type = BLOCK_TEXT;
    Block table1; table1.type = BLOCK_TABLE; table1.rows = 1; table1.cols = 1;
    Block form; form.type = BLOCK_FORM;
    Block table2; table2.type = BLOCK_TABLE; table2.rows = 2; table2.cols = 2;
    page.blocks = {text, table1, form, table2};

    std::vector<Block> tables = page.getTables();
    ASSERT_EQ((size_t)2, tables.size());
    EXPECT_EQ(1, tables[0].rows);
    EXPECT_EQ(2, tables[1].rows);
}

TEST(Docproc_Page, GetSentencesReturnsOneSentencePerLine)
{
    Page page;
    Line line1; line1.text = "Name: Jane Miller";
    Line line2; line2.text = "Invoice ID: INV-2048";
    Block block; block.type = BLOCK_TEXT; block.lines = {line1, line2};
    page.blocks = {block};

    std::vector<String> sentences = page.getSentences();
    ASSERT_EQ((size_t)2, sentences.size());
    EXPECT_EQ("Name: Jane Miller", sentences[0]);
    EXPECT_EQ("Invoice ID: INV-2048", sentences[1]);
}

TEST(Docproc_Page, GetSentencesSkipsEmptyLines)
{
    Page page;
    Line line1; line1.text = "First.";
    Line line2; line2.text = "";
    Line line3; line3.text = "Second.";
    Block block; block.type = BLOCK_TEXT; block.lines = {line1, line2, line3};
    page.blocks = {block};

    std::vector<String> sentences = page.getSentences();
    ASSERT_EQ((size_t)2, sentences.size());
    EXPECT_EQ("First.", sentences[0]);
    EXPECT_EQ("Second.", sentences[1]);
}

TEST(Docproc_Page, GetSentencesSkipsNonTextBearingBlocks)
{
    Page page;
    Line line; line.text = "Real sentence.";
    Block text; text.type = BLOCK_TEXT; text.lines = {line};

    Cell cell; cell.text = "Not a sentence. Should not appear.";
    Block table; table.type = BLOCK_TABLE; table.cells = {cell};

    page.blocks = {text, table};

    std::vector<String> sentences = page.getSentences();
    ASSERT_EQ((size_t)1, sentences.size());
    EXPECT_EQ("Real sentence.", sentences[0]);
}

}} // namespace opencv_test::(anonymous)
